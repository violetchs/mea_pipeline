
import matlab.engine
import os

class AxionSpkReader:
    """
    使用 MATLAB AxionFileLoader 读取 .spk 文件，需安装 matlab.engine 和 AxionFileLoader 工具箱。
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.eng = None

    def start_matlab(self):
        self.eng = matlab.engine.start_matlab()
        print("MATLAB 引擎已启动")

    def read_spk(self):
        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"文件不存在: {self.file_path}")
        if self.eng is None:
            self.start_matlab()
        self.eng.cd(os.path.dirname(self.file_path), nargout=0)
        # 加载 AxisFile 对象

        # 步骤1：加载 .spk 文件
        self.eng.eval(f"f = AxisFile('{self.file_path}');", nargout=0)
        self.eng.eval("Data = f.SpikeData.LoadData;", nargout=0)

        # 步骤2：遍历每个位置
        size = self.eng.eval("size(Data);", nargout=1)
        well_rows, well_cols, elec_cols, elec_rows = map(int, size._data)

        result = []

        for w_row in range(1, well_rows + 1):
            for w_col in range(1, well_cols + 1):
                well_label = chr(ord("A") + w_row - 1) + str(w_col)
                for e_col in range(1, elec_cols + 1):
                    for e_row in range(1, elec_rows + 1):
                        electrode_label = f"r{e_row}c{e_col}"
                        index_str = f"{{{w_row},{w_col},{e_col},{e_row}}}"

                        try:
                            is_empty = self.eng.eval(f"isempty(Data{index_str})", nargout=1)
                        except Exception as e:
                            print(f"[错误] 检查 {index_str} 是否为空失败：{e}")
                            is_empty = True

                        if is_empty:
                            result.append({
                                "well": well_label,
                                "electrode": electrode_label,
                                "data": {
                                    "spike_times": None,
                                    "spike_waveform": None
                                }
                            })
                            continue

                        try:
                            self.eng.eval(f"spikeTimes = [Data{index_str}(:).Start];", nargout=0)
                            self.eng.eval(f"spikeWaveform = Data{index_str}(1).GetVoltageVector;", nargout=0)

                            spike_times = self.eng.workspace["spikeTimes"]
                            spike_waveform = self.eng.workspace["spikeWaveform"]
                        except Exception as e:
                            print(f"[警告] 提取 {well_label}-{electrode_label} 数据失败: {e}")
                            spike_times = None
                            spike_waveform = None

                        result.append({
                            "well": well_label,
                            "electrode": electrode_label,
                            "data": {
                                "spike_times": spike_times,
                                "spike_waveform": spike_waveform
                            }
                        })
        return result

    def close(self):
        if self.eng:
            self.eng.quit()