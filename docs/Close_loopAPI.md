Closed-Loop API
Prerequisites
The static library you need to link your code against has been written in C++. To simplify the linking step, the interface of the library only exposes C-compatible types (i.e. only functions, struct’s, and no classes, etc.). It was compiled with:

Compiler: gcc-11.2.1
GLIBC: 2.3.2
CPP-Standard: GNU C++20 (-std=gnu++20)
Introduction
To conduct closed-loop, real-time experiments, a different approach is necessary. The general setup remains similar to the previous one, with an Electrode Array preparation and sequence creation. However, to gain more precise control over when to transmit the prepared sequence to the MaxOne/MaxTwo device, an additional C++ interface comes into play. This interface monitors the incoming data stream and swiftly delivers a response (the closed-loop sequence) based on predefined custom conditions. This bi-directional data exchange involves concurrent reading and transmission on two separate ports.

Note
Before starting with closed-loop experiments, it is highly recommended to be familiar with the Python API code explained above. Additionally, C++ knowledge is required. You will find all referenced files in this tutorial in the maxlab_lib folder, which you get by unzipping ~/MaxLab/share/libmaxlab-*.zip.
For this example, we will simulate a triggering sequence here. In a real experimental setup, this sequence should not be created, as the triggers will come from your cells.

Python Setup
The first step involves creating a python setup script, following the guidelines outlined in the Python API tutorial, with the following additional changes.

System Initialization

Firstly, two additional parameters need to be set. The first parameter defines the amplitude of the trigger stimulation sequence, which originates from an electrode named trigger_electrode. The second parameter defines the amplitude of the closed-loop response, generated from an electrode designated as closed_loop_electrode. The newly introduced parameters are as follows:

trigger_electrode = 13248
closed_loop_electrode = 13378
trigger_stimulation_amplitude = 6
closed_loop_stimulation_amplitude = 10

Note
When using the closed-loop scripts outside of this tutorial, there is no need to simulate a trigger sequence. The simulation is solely for illustrative purposes. Hence, there is no necessity to set trigger_stimulation_amplitude. Instead, you need to set a condition in the C++ script, as explained later.
Prior to initializing the system, you need to make sure that no sequence has been pre-defined in the server. This can be achieved with the following procedure:

s = mx.Sequence('trigger', persistent=False)
del(s)
s = mx.Sequence('closed_loop', persistent=False)
del(s)

The system is now ready for initialization, as detailed earlier. However, for the purposes of this example, a final adjustment is required to mitigate the occurrence of false positives. This adjustment involves setting the detection threshold to 8.5 times the standard deviation of the noise:

mx.set_event_threshold(8.5)

Array and Stimulation Units Preparation

The configuration process for the Electrode Array and Stimulation Units follows the same steps as detailed above. However, there is a key distinction: unlike in the Python API tutorial, where multiple electrodes are stimulated, here, only two electrodes will be stimulated. These two electrodes will be associated with their respective stimulation units, designated as stimulation_1 and stimulation_2.

Stimulation Sequence(s) Preparation

The stimulation sequences (in our case the trigger and closed-loop sequences) can be prepared similarly as explained in the Python API. The two must be associated with different DACs so that they can independently be controlled. Assuming that the function create_stim_pulse as defined above exists, we would run the following code:

sequence_1 = mx.Sequence('trigger', persistent=True)
for _ in range(stimulation_pulses_number):
    sequence_1 = create_stim_pulse(sequence_1, trigger_stimulation_amplitude, phase, 0)
    sequence_1.append(mx.DelaySamples(interpulse_interval))

sequence_2 = mx.Sequence('closed_loop_sequence', persistent=True)
for _ in range(stimulation_pulses_number):
    sequence_2 = create_stim_pulse(sequence_2, closed_loop_stimulation_amplitude, phase, 1)
    sequence_2.append(mx.DelaySamples(interpulse_interval))

Similar to the explanation provided in the Python API Tutorial, to this point, no data has been sent to the device, and the Electrode Array and sequence preparations have been executed only locally.

Stimulation Sequence(s) Sending and Data Saving

Sending and recording the data must be then run similarly as in the Python API tutorial.

Once the script is ready, it should not be executed immediately. It should be saved with a filename like closedLoopSetup.py, and run from a terminal or executable after setting up the C++ part.

C++ Setup
This code interacts with the MaxOne/MaxTwo, listening to the data stream and responding in real time. There are two distinct types of data streams: the raw data stream and the filtered data stream. The former contains unprocessed data, while the latter exclusively includes the spikes detected by MaxLab Live.

Raw Data Stream

We start here by explaining how the C++ code for processing raw data works. Here is a copy of the script:

#include <stdlib.h>
#include <stdio.h>
#include <thread>
#include <chrono>
#include "maxlab/maxlab.h"

int main(int argc, char * argv[])
{
    if (argc < 2)
    {
        fprintf(stderr, "Call with: %s [detection_channel]", argv[0]);
        exit(1);
    }
    const int detection_channel = atoi(argv[1]);

    maxlab::checkVersions();
    maxlab::verifyStatus(maxlab::DataStreamerRaw_open());
    std::this_thread::sleep_for(std::chrono::seconds(2));//Allow data stream to open

    uint64_t blanking = 0;
    while (true)
    {
        maxlab::RawFrameData frameData;
        maxlab::Status status = maxlab::DataStreamerRaw_receiveNextFrame(&frameData);
        if (status == maxlab::Status::MAXLAB_NO_FRAME || frameData.frameInfo.corrupted)
            continue;
        if (blanking > 0)
        {
            blanking--;
            if(blanking != 0)
                continue;
        }

        if (frameData.amplitudes[detection_channel] > 40.f) //Amplitudes can be variable. Adjust this as necessary.
        {
            maxlab::verifyStatus(maxlab::sendSequence("closed_loop"));
            blanking = 8000;
        }
    }
    maxlab::verifyStatus(maxlab::DataStreamerRaw_close());
}

The general approach of the code is the following: we first open the stream from which we will receive real-time data that is coming from the MaxOne/MaxTwo. Notice how in the end we need to again close this stream. This is important because otherwise we will leave the mxwserver in a undefined state! We also call the function maxlab::verifyStatus to ensure that no errors happened during the opening of the stream. You should only call this function if the error is not recoverable, since we exit execution in case something goes wrong. The next step is to create an instance of maxlab::RawFrameData in which the information that is received in maxlab::DataStreamerRaw_receiveNextFrame can be stored in. Notice that we also check the status of the reception to make sure that actually received data (the while-loop might be faster than the sampling rate of the device and we thus might not have any data in the stream). If we have data, we can check a condition to decide if we want to stimulate or not. Here we have chosen to simply stimulate if we have a spike on a particular channel (i.e. detection_channel). In that case we call maxlab::sendSequence to send the response stimulus.

Most of the general structure from the example above can be taken for any experiment. A simple generalization here might be to write a function with the signature bool condition(maxlab::RawFrameData&) that takes the received data and returns a bool based on the input if a stimulation pulse should be sent.

if (condition(frameData))
{
    maxlab::verifyStatus(maxlab::sendSequence("closed_loop"));
    blanking = 8000;
}

To complete our understanding of this code, we here briefly explain the role that the variable blanking plays. After sending a stimulation pulse to the MaxOne/MaxTwo, the voltage of the chip needs some amount of time until returning back to neutral (0V). This is done by continuously decrementing this variable after having sent the stimulation pulse.

Note
Since the blanking variable is decremented for every received frame, this variable needs to be set differently for MaxOne/MaxTwo with their different sampling rates. In this example, the 8000 frames would correspond to 0.4 ms and 0.8 ms for MaxOne/MaxTwo respectively. Additionally, 0.4 ms is a very conservative number and can likely be decreased further without any issues.
Filtered Data Stream

The process_spike_events.cpp (filtered data stream) script closely resembles the raw data stream script. However, it distinguishes itself through the incorporation of an additional for-loop. For every frame, this loop checks if any spikes were detected on the detection channel, triggering the stimulation if that was the case.

#include <iostream>
#include <thread>
#include <chrono>

#include "maxlab/maxlab.h"
#include <unistd.h>

int main(int argc, char * argv[])
{
    if (argc < 2)
    {
        std::cerr << "Call with:\t" << argv[0] << "\t[detection_channel]" << std::endl;
        exit(1);
    }
    const int detection_channel = atoi(argv[1]);

    uint8_t targetWell{0};

    uint64_t blanking{0};
    maxlab::checkVersions();
    maxlab::verifyStatus(maxlab::DataStreamerFiltered_open(maxlab::FilterType::IIR));
    std::this_thread::sleep_for(std::chrono::seconds(2));;//Allow data stream to open

    maxlab::FilteredFrameData frameData;
    while (true)
    {
        maxlab::Status status = maxlab::DataStreamerFiltered_receiveNextFrame(&frameData);
        if (status == maxlab::Status::MAXLAB_NO_FRAME)
            continue;

        if (frameData.frameInfo.well_id != targetWell)
            continue;

        if (blanking > 0)
        {
            blanking--;
            continue;
        }

        for (uint64_t i = 0; i < frameData.spikeCount; ++i)
        {
            const maxlab::SpikeEvent & spike = frameData.spikeEvents[i];
            if (spike.channel == detection_channel)
            {
                maxlab::verifyStatus(maxlab::sendSequence("closed_loop"));
                blanking = 8000;
            }
        }
    }
    maxlab::verifyStatus(maxlab::DataStreamerFiltered_close());
}

This ensures that only actual spikes are processed and what doesn’t pass as a spike event is not considered. For more details, have a look at maxlab::SpikeEvent.

Integrating everything together
Upon properly setting up the scripts, you can start the experiment. This process follows a specific sequence. Begin by compiling the C++ code; execute this step by running the make command from the main folder. This will generate a new directory named build, where the executable file can be found:

# cd /path/to/maxlab_lib
make

For more information on GNU CMake, please consult the official documentation.

Once the executable is created, the mxwserver can be launched through the GUI. The next step will be to run the compiled C++ script:

# replace `detection_channel` with the appropriate number
./example_raw detection_channel

And, finally, we are ready to run the Python setup script, called here closeLoopSetup.py. This can be done with:

python3 closeLoopSetup.py

Note
To execute the Python script, it is important to install the maxlab package first. If a virtual environment, such as pyenv or conda, is in use, ensure that the environment is activated correctly.
The experiment is in progress, and it can be observed by accessing the GUI. The data is systematically saved.

Other experimental workflows are also possible. Another option is to create a Python setup script that calls the C++ executable from within, so that only the Python script needs to be manually run.



注意事项： The command mx.clear_events() is being called after the event is defined, which clears the event again. This command needs to be used before defining the events.
