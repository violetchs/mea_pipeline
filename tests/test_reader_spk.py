
from src.pipeline.io.reader_spk import AxionSpkReader

def test_read_spk():
    example_path = r"F:\20250611\20250611-H8-spon0(000)(000).spk"
    reader = AxionSpkReader(example_path)
    try:
        spikes = reader.read_spk()
        print("MATLAB AxionFileLoader 成功读取数据")

    except FileNotFoundError:
        print(f"文件未找到: {example_path}（测试跳过）")

    except Exception as e:
        print(f"读取失败: {str(e)}")

    reader.close()

if __name__ == "__main__":
    test_read_spk()
