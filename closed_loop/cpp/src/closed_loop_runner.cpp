#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cctype>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "maxlab/maxlab.h"

namespace {

std::atomic<bool> g_stop{false};

void on_signal(int)
{
    g_stop.store(true);
}

const char *status_to_string(maxlab::Status st)
{
    switch (st)
    {
    case maxlab::Status::MAXLAB_OK:
        return "MAXLAB_OK";
    case maxlab::Status::MAXLAB_API_FAIL:
        return "MAXLAB_API_FAIL";
    case maxlab::Status::MAXLAB_NO_SERVER_CONNECTION:
        return "MAXLAB_NO_SERVER_CONNECTION";
    case maxlab::Status::MAXLAB_LICENSE_INVALID:
        return "MAXLAB_LICENSE_INVALID";
    case maxlab::Status::MAXLAB_API_NO_RESPONSE:
        return "MAXLAB_API_NO_RESPONSE";
    case maxlab::Status::MAXLAB_STREAM_ALREADY_OPENED:
        return "MAXLAB_STREAM_ALREADY_OPENED";
    case maxlab::Status::MAXLAB_INCOMPATIBLE_FILTERING:
        return "MAXLAB_INCOMPATIBLE_FILTERING";
    case maxlab::Status::MAXLAB_INVALID_INPUT:
        return "MAXLAB_INVALID_INPUT";
    case maxlab::Status::MAXLAB_STREAM_NOT_OPENED:
        return "MAXLAB_STREAM_NOT_OPENED";
    case maxlab::Status::MAXLAB_NO_FRAME:
        return "MAXLAB_NO_FRAME";
    default:
        return "unknown Status";
    }
}

struct AutoSpec
{
    bool enabled = false;
    std::string mode = "top";
    int count = 8;
    double window_s = 5.0;
    double rate_threshold_hz = 0.0;
};

struct Rule
{
    double start_s = 0.0;
    double stop_s = -1.0;
    std::string detect_spec;
    AutoSpec auto_spec;
    std::vector<int> detect_channels;
    int threshold = 1;
    std::string sequence = "closed_loop";
};

struct PendingArtifact
{
    bool active = true;
    unsigned long command_frame = 0;
    unsigned long deadline_frame = 0;
    double command_time_s = 0.0;
    int stim_count = 0;
    std::string sequence;
    std::vector<int> channels;
};

struct Args
{
    std::string rules_json;
    std::string rules_file;
    std::string detection_channel;
    std::string sequence = "closed_loop";
    int threshold = 1;
    double blank_ms = 500.0;
    double run_seconds = 0.0;
    double sample_rate_hz = 20000.0;
    double artifact_window_ms = 25.0;
    double telemetry_interval_ms = 250.0;
    int telemetry_max_events = 1024;
};

std::string trim(const std::string &value)
{
    size_t start = 0;
    while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start])))
        ++start;
    size_t stop = value.size();
    while (stop > start && std::isspace(static_cast<unsigned char>(value[stop - 1])))
        --stop;
    return value.substr(start, stop - start);
}

bool starts_with(const std::string &text, const std::string &prefix)
{
    return text.rfind(prefix, 0) == 0;
}

std::vector<std::string> split_tokens(const std::string &text)
{
    std::vector<std::string> out;
    std::string token;
    for (char ch : text)
    {
        if (ch == ',' || ch == ';' || std::isspace(static_cast<unsigned char>(ch)))
        {
            if (!token.empty())
            {
                out.push_back(token);
                token.clear();
            }
        }
        else
        {
            token.push_back(ch);
        }
    }
    if (!token.empty())
        out.push_back(token);
    return out;
}

std::vector<int> parse_channels(const std::string &text)
{
    std::vector<int> channels;
    for (const auto &token : split_tokens(text))
    {
        if (token == "all")
        {
            channels.clear();
            for (int i = 0; i < 1024; ++i)
                channels.push_back(i);
            return channels;
        }
        const auto dash = token.find('-');
        if (dash != std::string::npos)
        {
            int start = std::atoi(token.substr(0, dash).c_str());
            int stop = std::atoi(token.substr(dash + 1).c_str());
            int step = stop >= start ? 1 : -1;
            for (int value = start;; value += step)
            {
                if (value >= 0 && value < 1024)
                    channels.push_back(value);
                if (value == stop)
                    break;
            }
            continue;
        }
        int channel = std::atoi(token.c_str());
        if (channel >= 0 && channel < 1024)
            channels.push_back(channel);
    }
    std::sort(channels.begin(), channels.end());
    channels.erase(std::unique(channels.begin(), channels.end()), channels.end());
    return channels;
}

AutoSpec parse_auto_spec(const std::string &text)
{
    AutoSpec spec;
    if (!starts_with(text, "auto"))
        return spec;
    spec.enabled = true;
    std::vector<std::string> parts;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ':'))
    {
        item = trim(item);
        if (!item.empty())
            parts.push_back(item);
    }
    if (parts.size() > 1)
        spec.mode = parts[1];
    if (spec.mode == "active" || spec.mode == "highest" || spec.mode == "high" || spec.mode == "rate")
        spec.mode = "top";
    if (spec.mode == "quiet" || spec.mode == "lowest" || spec.mode == "low")
        spec.mode = "bottom";
    if (spec.mode != "top" && spec.mode != "bottom" && spec.mode != "above")
        spec.mode = "top";
    if (spec.mode == "above")
    {
        if (parts.size() > 2)
            spec.rate_threshold_hz = std::max(0.0, std::atof(parts[2].c_str()));
        if (parts.size() > 3)
            spec.window_s = std::max(0.25, std::atof(parts[3].c_str()));
    }
    else
    {
        if (parts.size() > 2)
            spec.count = std::max(1, std::atoi(parts[2].c_str()));
        if (parts.size() > 3)
            spec.window_s = std::max(0.25, std::atof(parts[3].c_str()));
    }
    return spec;
}

std::string read_file(const std::string &path)
{
    std::FILE *fp = std::fopen(path.c_str(), "rb");
    if (!fp)
        return "";
    std::string data;
    char buffer[4096];
    while (true)
    {
        const size_t n = std::fread(buffer, 1, sizeof(buffer), fp);
        if (n > 0)
            data.append(buffer, n);
        if (n < sizeof(buffer))
            break;
    }
    std::fclose(fp);
    return data;
}

std::string json_string_value(const std::string &object, const std::string &key, const std::string &fallback = "")
{
    std::regex pattern("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
    std::smatch match;
    if (std::regex_search(object, match, pattern))
        return match[1].str();
    return fallback;
}

double json_number_value(const std::string &object, const std::string &key, double fallback)
{
    std::regex pattern("\"" + key + "\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?)");
    std::smatch match;
    if (std::regex_search(object, match, pattern))
        return std::atof(match[1].str().c_str());
    return fallback;
}

std::vector<int> json_int_array(const std::string &object, const std::string &key)
{
    std::vector<int> result;
    std::regex pattern("\"" + key + "\"\\s*:\\s*\\[([^\\]]*)\\]");
    std::smatch match;
    if (!std::regex_search(object, match, pattern))
        return result;
    std::regex value_pattern("\"?(-?[0-9]+)\"?");
    std::string body = match[1].str();
    for (std::sregex_iterator it(body.begin(), body.end(), value_pattern), end; it != end; ++it)
    {
        const int channel = std::atoi((*it)[1].str().c_str());
        if (channel >= 0 && channel < 1024)
            result.push_back(channel);
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

std::vector<std::string> json_objects(const std::string &json)
{
    std::vector<std::string> objects;
    int depth = 0;
    bool in_string = false;
    bool escape = false;
    size_t start = std::string::npos;
    for (size_t i = 0; i < json.size(); ++i)
    {
        const char ch = json[i];
        if (escape)
        {
            escape = false;
            continue;
        }
        if (ch == '\\' && in_string)
        {
            escape = true;
            continue;
        }
        if (ch == '"')
        {
            in_string = !in_string;
            continue;
        }
        if (in_string)
            continue;
        if (ch == '{')
        {
            if (depth == 0)
                start = i;
            ++depth;
        }
        else if (ch == '}')
        {
            --depth;
            if (depth == 0 && start != std::string::npos)
            {
                objects.push_back(json.substr(start, i - start + 1));
                start = std::string::npos;
            }
        }
    }
    return objects;
}

std::vector<Rule> parse_rules(const Args &args)
{
    std::string json = args.rules_json;
    if (json.empty() && !args.rules_file.empty())
        json = read_file(args.rules_file);
    std::vector<Rule> rules;
    for (const auto &object : json_objects(json))
    {
        Rule rule;
        rule.start_s = json_number_value(object, "start_s", 0.0);
        rule.stop_s = json_number_value(object, "stop_s", -1.0);
        rule.threshold = std::max(1, static_cast<int>(json_number_value(object, "threshold", 1.0)));
        rule.sequence = json_string_value(object, "sequence", "closed_loop");
        rule.detect_spec = json_string_value(object, "detect_spec", "");
        if (rule.detect_spec.empty())
            rule.detect_spec = json_string_value(object, "detect", "");
        rule.auto_spec = parse_auto_spec(rule.detect_spec);
        rule.detect_channels = json_int_array(object, "detect_channels");
        if (!rule.auto_spec.enabled && rule.detect_channels.empty() && !rule.detect_spec.empty())
            rule.detect_channels = parse_channels(rule.detect_spec);
        rules.push_back(rule);
    }
    if (rules.empty())
    {
        Rule rule;
        rule.sequence = args.sequence;
        rule.threshold = std::max(1, args.threshold);
        rule.detect_spec = args.detection_channel;
        rule.auto_spec = parse_auto_spec(rule.detect_spec);
        rule.detect_channels = rule.auto_spec.enabled ? std::vector<int>() : parse_channels(args.detection_channel);
        if (rule.detect_channels.empty() && !rule.auto_spec.enabled)
            rule.detect_channels.push_back(0);
        rules.push_back(rule);
    }
    return rules;
}

Args parse_args(int argc, char **argv)
{
    Args args;
    for (int i = 1; i < argc; ++i)
    {
        std::string key = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc)
                return "";
            return argv[++i];
        };
        if (key == "--rules")
            args.rules_json = next();
        else if (key == "--rules-file")
            args.rules_file = next();
        else if (key == "--detection-channel" || key == "--detection-channels")
            args.detection_channel = next();
        else if (key == "--sequence")
            args.sequence = next();
        else if (key == "--threshold-spikes")
            args.threshold = std::atoi(next().c_str());
        else if (key == "--blank-ms")
            args.blank_ms = std::atof(next().c_str());
        else if (key == "--run-seconds")
            args.run_seconds = std::atof(next().c_str());
        else if (key == "--sample-rate-hz")
            args.sample_rate_hz = std::atof(next().c_str());
        else if (key == "--artifact-window-ms")
            args.artifact_window_ms = std::atof(next().c_str());
        else if (key == "--telemetry-interval-ms")
            args.telemetry_interval_ms = std::atof(next().c_str());
        else if (key == "--telemetry-max-events")
            args.telemetry_max_events = std::atoi(next().c_str());
        else if (key == "--cfg" || key == "--response-electrode" || key == "--amplitude-mv")
            (void)next();
    }
    return args;
}

std::vector<int> auto_channels(const AutoSpec &spec, const std::vector<std::deque<unsigned long>> &spikes, unsigned long frame_no, double sample_rate_hz)
{
    std::vector<std::pair<double, int>> scored;
    const double window_s = std::max(0.25, spec.window_s);
    const double denom = std::max(1e-9, window_s);
    for (int channel = 0; channel < 1024; ++channel)
    {
        const double rate = static_cast<double>(spikes[channel].size()) / denom;
        if (spec.mode == "above")
        {
            if (rate >= spec.rate_threshold_hz)
                scored.push_back({-rate, channel});
        }
        else if (spec.mode == "bottom")
        {
            scored.push_back({rate, channel});
        }
        else
        {
            scored.push_back({-rate, channel});
        }
    }
    std::sort(scored.begin(), scored.end());
    std::vector<int> result;
    const int limit = spec.mode == "above" ? static_cast<int>(scored.size()) : std::min(spec.count, static_cast<int>(scored.size()));
    for (int i = 0; i < limit; ++i)
        result.push_back(scored[i].second);
    (void)frame_no;
    (void)sample_rate_hz;
    return result;
}

std::vector<int> resolved_channels(const Rule &rule, const std::vector<std::deque<unsigned long>> &spikes, unsigned long frame_no, double sample_rate_hz)
{
    if (rule.auto_spec.enabled)
        return auto_channels(rule.auto_spec, spikes, frame_no, sample_rate_hz);
    return rule.detect_channels;
}

void log_event(double time_s, const std::string &type, const std::string &channel, int count, const std::string &note, double latency_ms = -1.0, double dispatch_ms = -1.0)
{
    std::cout << "{\"time_s\":" << time_s
              << ",\"type\":\"" << type
              << "\",\"channel\":\"" << channel
              << "\",\"count\":" << count
              << ",\"note\":\"" << note << "\"";
    if (latency_ms >= 0.0)
        std::cout << ",\"latency_ms\":" << latency_ms;
    if (dispatch_ms >= 0.0)
        std::cout << ",\"dispatch_ms\":" << dispatch_ms;
    std::cout << "}" << std::endl;
}

void log_spike_batch(double time_s, const std::vector<std::pair<int, double>> &events)
{
    if (events.empty())
        return;
    std::cout << "{\"time_s\":" << time_s
              << ",\"type\":\"spikes\",\"channels\":[";
    for (size_t i = 0; i < events.size(); ++i)
    {
        if (i)
            std::cout << ",";
        std::cout << events[i].first;
    }
    std::cout << "],\"times_s\":[";
    for (size_t i = 0; i < events.size(); ++i)
    {
        if (i)
            std::cout << ",";
        std::cout << events[i].second;
    }
    std::cout << "]}" << std::endl;
}

bool contains_channel(const std::vector<int> &channels, int channel)
{
    return channels.empty() || std::find(channels.begin(), channels.end(), channel) != channels.end();
}

std::string join_channels(const std::vector<int> &channels, size_t max_items = 16)
{
    std::ostringstream oss;
    for (size_t i = 0; i < channels.size() && i < max_items; ++i)
    {
        if (i)
            oss << ",";
        oss << channels[i];
    }
    if (channels.size() > max_items)
        oss << "...";
    return oss.str();
}

} // namespace

int main(int argc, char **argv)
{
    Args args = parse_args(argc, argv);
    std::vector<Rule> rules = parse_rules(args);
    if (rules.empty())
    {
        std::cerr << "No closed-loop rules configured." << std::endl;
        return 1;
    }

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    maxlab::checkVersions();
    maxlab::verifyStatus(maxlab::DataStreamerFiltered_open(maxlab::FilterType::IIR));
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    std::vector<std::deque<unsigned long>> spikes(1024);
    const auto t0 = std::chrono::steady_clock::now();
    auto blank_until = t0;
    int stim_count = 0;
    std::vector<PendingArtifact> pending_artifacts;
    std::vector<std::pair<int, double>> telemetry_events;
    auto last_telemetry_flush = t0;
    unsigned long stream_frame0 = 0;
    bool have_stream_frame0 = false;

    std::cout << "closed_loop_runner started with " << rules.size() << " rule(s)" << std::endl;
    for (size_t i = 0; i < rules.size(); ++i)
    {
        std::cout << "rule " << (i + 1) << ": sequence=" << rules[i].sequence
                  << " threshold=" << rules[i].threshold
                  << " detect=" << (rules[i].detect_spec.empty() ? join_channels(rules[i].detect_channels) : rules[i].detect_spec)
                  << std::endl;
    }

    while (!g_stop.load())
    {
        const auto now = std::chrono::steady_clock::now();
        const double elapsed_s = std::chrono::duration<double>(now - t0).count();
        if (args.run_seconds > 0.0 && elapsed_s >= args.run_seconds)
            break;

        maxlab::FilteredFrameData frame{};
        const maxlab::Status status = maxlab::DataStreamerFiltered_receiveNextFrame(&frame);
        if (status == maxlab::Status::MAXLAB_NO_FRAME)
            continue;
        if (status != maxlab::Status::MAXLAB_OK)
        {
            std::cerr << "DataStreamerFiltered_receiveNextFrame failed: " << status_to_string(status) << std::endl;
            break;
        }
        if (frame.frameInfo.corrupted)
            continue;

        unsigned long frame_no = static_cast<unsigned long>(frame.frameInfo.frame_number);
        if (!have_stream_frame0)
        {
            stream_frame0 = frame_no;
            have_stream_frame0 = true;
        }
        for (uint64_t i = 0; i < frame.spikeCount; ++i)
        {
            const auto &event = frame.spikeEvents[i];
            if (event.channel < 1024)
            {
                spikes[event.channel].push_back(event.frameNo);
                frame_no = std::max<unsigned long>(frame_no, event.frameNo);
                if (static_cast<int>(telemetry_events.size()) < std::max(1, args.telemetry_max_events))
                {
                    const double event_time_s = event.frameNo >= stream_frame0
                                                    ? static_cast<double>(event.frameNo - stream_frame0) / args.sample_rate_hz
                                                    : elapsed_s;
                    telemetry_events.push_back({static_cast<int>(event.channel), event_time_s});
                }
            }
        }

        for (auto &pending : pending_artifacts)
        {
            if (!pending.active)
                continue;
            bool found = false;
            for (uint64_t i = 0; i < frame.spikeCount; ++i)
            {
                const auto &event = frame.spikeEvents[i];
                if (event.frameNo <= pending.command_frame || event.frameNo > pending.deadline_frame)
                    continue;
                if (!contains_channel(pending.channels, static_cast<int>(event.channel)))
                    continue;
                const double latency_ms = (static_cast<double>(event.frameNo - pending.command_frame) / args.sample_rate_hz) * 1000.0;
                const double artifact_time_s = pending.command_time_s + latency_ms / 1000.0;
                log_event(artifact_time_s, "artifact", std::to_string(event.channel), pending.stim_count, pending.sequence, latency_ms);
                pending.active = false;
                found = true;
                break;
            }
            if (!found && frame_no > pending.deadline_frame)
            {
                const double window_ms = (static_cast<double>(pending.deadline_frame - pending.command_frame) / args.sample_rate_hz) * 1000.0;
                log_event(pending.command_time_s + window_ms / 1000.0, "artifact_miss", pending.sequence, pending.stim_count, "no filtered artifact", -1.0);
                pending.active = false;
            }
        }
        pending_artifacts.erase(
            std::remove_if(pending_artifacts.begin(), pending_artifacts.end(), [](const PendingArtifact &item) { return !item.active; }),
            pending_artifacts.end());

        if (args.telemetry_interval_ms > 0.0)
        {
            const double since_flush_ms = std::chrono::duration<double, std::milli>(now - last_telemetry_flush).count();
            if (since_flush_ms >= args.telemetry_interval_ms)
            {
                log_spike_batch(elapsed_s, telemetry_events);
                telemetry_events.clear();
                last_telemetry_flush = now;
            }
        }

        double max_window_s = 5.0;
        for (const auto &rule : rules)
        {
            if (rule.auto_spec.enabled)
                max_window_s = std::max(max_window_s, rule.auto_spec.window_s);
        }
        const unsigned long keep_after = frame_no > static_cast<unsigned long>(max_window_s * args.sample_rate_hz)
                                             ? frame_no - static_cast<unsigned long>(max_window_s * args.sample_rate_hz)
                                             : 0;
        for (auto &queue : spikes)
        {
            while (!queue.empty() && queue.front() < keep_after)
                queue.pop_front();
        }

        if (now < blank_until)
            continue;

        for (const auto &rule : rules)
        {
            if (elapsed_s < rule.start_s)
                continue;
            if (rule.stop_s >= 0.0 && elapsed_s > rule.stop_s)
                continue;
            const auto channels = resolved_channels(rule, spikes, frame_no, args.sample_rate_hz);
            int count = 0;
            for (int channel : channels)
            {
                if (channel >= 0 && channel < 1024)
                    count += static_cast<int>(spikes[channel].size());
            }
            if (count < rule.threshold)
                continue;

            const auto command_start = std::chrono::steady_clock::now();
            const maxlab::Status send_status = maxlab::sendSequence(rule.sequence.c_str());
            const auto command_done = std::chrono::steady_clock::now();
            const double command_start_s = std::chrono::duration<double>(command_start - t0).count();
            const double command_done_s = std::chrono::duration<double>(command_done - t0).count();
            const double dispatch_latency_ms = std::chrono::duration<double, std::milli>(command_done - command_start).count();
            if (send_status != maxlab::Status::MAXLAB_OK)
            {
                std::cerr << "sendSequence(" << rule.sequence << ") failed: " << status_to_string(send_status) << std::endl;
                maxlab::verifyStatus(maxlab::DataStreamerFiltered_close());
                return 2;
            }
            ++stim_count;
            log_event(command_start_s, "trigger", join_channels(channels), count, rule.detect_spec.empty() ? "threshold" : rule.detect_spec);
            log_event(command_done_s, "stim", rule.sequence, stim_count, "sendSequence_ack", -1.0, dispatch_latency_ms);
            PendingArtifact pending;
            pending.command_frame = frame_no;
            pending.deadline_frame = frame_no + static_cast<unsigned long>(std::max(1.0, args.artifact_window_ms / 1000.0 * args.sample_rate_hz));
            pending.command_time_s = command_start_s;
            pending.stim_count = stim_count;
            pending.sequence = rule.sequence;
            pending.channels = channels;
            pending_artifacts.push_back(pending);
            blank_until = now + std::chrono::milliseconds(static_cast<int>(std::max(0.0, args.blank_ms)));
            break;
        }
    }

    maxlab::verifyStatus(maxlab::DataStreamerFiltered_close());
    std::cout << "closed_loop_runner stopped; stim_count=" << stim_count << std::endl;
    return 0;
}
