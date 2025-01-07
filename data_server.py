import os
import pickle
import glob
from torch.utils.data import Dataset
import zmq

# ===== 数据加载服务端 =====
def data_loader_server(train_pattern=".discard_model/train_*.pkl", val_pattern=".discard_model/valid_*.pkl", address="tcp://*:5555"):
    """
    数据加载服务端，预加载数据并通过 ZeroMQ 提供接口。
    Args:
        train_pattern (str): 训练数据文件路径的模式。
        val_pattern (str): 验证数据文件路径的模式。
        address (str): ZeroMQ 服务绑定地址。
    """
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(address)

    # 预加载并序列化训练数据
    print("[Server] Loading and serializing training data...")
    train_files = sorted(glob.glob(train_pattern))
    train_data = []
    for file in train_files:
        with open(file, 'rb') as f:
            data = pickle.load(f)
            train_data.extend(zip(data[0], data[1], data[2]))
    train_data_serialized = [pickle.dumps(item) for item in train_data]

    # 预加载并序列化验证数据
    print("[Server] Loading and serializing validation data...")
    val_files = sorted(glob.glob(val_pattern))
    val_data = []
    for file in val_files:
        with open(file, 'rb') as f:
            data = pickle.load(f)
            val_data.extend(zip(data[0], data[1], data[2]))
    val_data_serialized = [pickle.dumps(item) for item in val_data]

    print("[Server] Data serialized. Ready to serve requests.")

    # 处理请求
    while True:
        message = socket.recv()
        request_type, dataset_type, index = pickle.loads(message)
        if request_type == "get":
            if dataset_type == "train":
                if 0 <= index < len(train_data_serialized):
                    socket.send(train_data_serialized[index])
                else:
                    socket.send(b"")  # Index out of range
            elif dataset_type == "val":
                if 0 <= index < len(val_data_serialized):
                    socket.send(val_data_serialized[index])
                else:
                    socket.send(b"")  # Index out of range
            else:
                socket.send(b"")  # Unknown dataset type
        elif request_type == "size":
            if dataset_type == "train":
                socket.send(pickle.dumps(len(train_data_serialized)))
            elif dataset_type == "val":
                socket.send(pickle.dumps(len(val_data_serialized)))
            else:
                socket.send(b"")  # Unknown dataset type
        elif request_type == "close":
            print("[Server] Shutting down.")
            break
        else:
            socket.send(b"")  # Unknown request type

# ===== 数据集客户端 =====
class MyDataset(Dataset):
    def __init__(self, dataset_type, address="tcp://localhost:5555", cached=False):
        """
        数据集客户端，通过 ZeroMQ 接口获取数据。
        Args:
            dataset_type (str): 数据集类型（"train" 或 "val"）。
            address (str): ZeroMQ 服务地址。
            cached (bool): 如果为 True，则将数据全部加载到内存中。
        """
        self.dataset_type = dataset_type
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(address)
        self.total_size = self._get_size()
        self.cached = cached

        if self.cached:
            self.cached_data = self._load_all_data()

    def _get_size(self):
        self.socket.send(pickle.dumps(("size", self.dataset_type, -1)))
        response = self.socket.recv()
        size = pickle.loads(response)
        if size is not None:
            return size
        else:
            raise ValueError("Failed to retrieve dataset size")

    def _load_all_data(self):
        """
        加载所有数据到内存中。
        """
        data = []
        for index in range(self.total_size):
            self.socket.send(pickle.dumps(("get", self.dataset_type, index)))
            response = self.socket.recv()
            if response:
                data.append(pickle.loads(response))
            else:
                raise ValueError(f"Failed to retrieve data at index {index}")
        return data

    def __len__(self):
        return self.total_size

    def __getitem__(self, index):
        if self.cached:
            return self.cached_data[index]
        else:
            self.socket.send(pickle.dumps(("get", self.dataset_type, index)))
            response = self.socket.recv()
            if response:
                return pickle.loads(response)
            else:
                raise IndexError("Index out of range or unknown error")

class MyLocalDataset(Dataset):
    def __init__(self, dataset_type):
        """
        本地数据集加载。
        Args:
            dataset_type (str): 数据集类型（"train" 或 "val"）。
        """
        if dataset_type == "train":
            self.file_pattern = ".discard_model/train_*.pkl"
        elif dataset_type == "val":
            self.file_pattern = ".discard_model/valid_*.pkl"
        else:
            raise ValueError("Invalid dataset type. Use 'train' or 'val'.")

        self.files = sorted(glob.glob(self.file_pattern))
        self.file_index_map = self._build_file_index_map()
        self.total_size = sum(len(indices) for indices in self.file_index_map.values())

    def _build_file_index_map(self):
        """
        构建文件到索引的映射，用于按需加载数据。
        Returns:
            dict: 文件路径到数据索引范围的映射。
        """
        file_index_map = {}
        start_index = 0
        for file in self.files:
            with open(file, 'rb') as f:
                data = pickle.load(f)
                data_length = len(data[1])  # 使用标签的长度作为数据长度
                file_index_map[file] = range(start_index, start_index + data_length)
                start_index += data_length
        return file_index_map

    def __len__(self):
        return self.total_size

    def __getitem__(self, index):
        for file, indices in self.file_index_map.items():
            if index in indices:
                local_index = index - indices.start
                with open(file, 'rb') as f:
                    data = pickle.load(f)
                    return (
                        data[0][local_index],  # 输入特征
                        data[1][local_index],  # 标签
                        data[2][local_index]   # 权重
                    )
        raise IndexError("Index out of range")

# ===== 启动数据加载服务端 =====
if __name__ == "__main__":
    data_loader_server()
