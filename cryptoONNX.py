import onnx
import onnx_graphsurgeon as gs
import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import hmac
import hashlib

class ONNXEncryptor:
    def __init__(self, key_hex: str, hmac_key_hex: str, nonce_hex: str):
        self.key = bytes.fromhex(key_hex)
        self.hmac_key = bytes.fromhex(hmac_key_hex)
        self.nonce = bytes.fromhex(nonce_hex)
        self.backend = default_backend()

    def _encrypt_weights(self, data_numpy: np.ndarray) -> np.ndarray:
        raw_bytes = data_numpy.tobytes()
        cipher = Cipher(algorithms.AES(self.key), modes.CTR(self.nonce), backend=self.backend)
        encryptor = cipher.encryptor()
        encrypted_data = encryptor.update(raw_bytes) + encryptor.finalize()

        return np.frombuffer(encrypted_data, dtype=np.uint8)

    def process_graph(self, graph: gs.Graph) -> gs.Graph:
        """
        Prende un oggetto gs.Graph, cifra i pesi e aggiunge come attributo ai nodi HMAAC sui pesi.
        """
        for node in graph.nodes:
            if len(node.inputs) > 1:
                print(f"Processing node {node.name}")
                for input in node.inputs:
                    print(input.name)
                    if isinstance(input, gs.Constant):
                        
                        # Weights 
                        if "weight_tensor" in input.name:
                            shape = input.values.shape
                            weights_tensor = np.array(input.values,dtype = np.int8)
                            # weights_tensor = weights_tensor.transpose(0,2,3,1)

                            # for i, byte in enumerate(weights_tensor.flatten()):
                            #     print(str(byte),end="\t")
                            
                            #     if (i + 1) % 8 == 0:
                            #         print(",")  
                            #     else:
                            #         print(", ",end="") 
                            encrypted_bytes = self._encrypt_weights(weights_tensor)
                            encrypted_bytes = encrypted_bytes.reshape(weights_tensor.shape)
                            digest = hmac.new(self.hmac_key, np.ascontiguousarray(encrypted_bytes), hashlib.sha256).digest()
                            hmac_array = np.frombuffer(digest,dtype=np.uint32)
                            # input.values = np.array(encrypted_bytes.transpose(0,3,1,2),dtype = np.int8)
                            input.values = np.array(encrypted_bytes,dtype = np.int8)
                            input.values = input.values.reshape(shape)
                            node.attrs['hmac'] = (gs.Constant(
                                name = f"{node.name}_hmac",
                                values = hmac_array
                            ))

                        # MatMul constants
                        elif "MatMul" in input.name or (hasattr(input, 'name') and "fc" in input.name):
                            og_shape = input.values.shape
                            weights_tensor = np.array(input.values, dtype=np.int8).transpose(1, 0)
                            
                            encrypted_bytes = self._encrypt_weights(weights_tensor)
                            encrypted_bytes.shape = weights_tensor.shape
                            
                            input.values = np.array(encrypted_bytes.transpose(1, 0), dtype=np.int8).reshape(og_shape)

                        # Biases
                        elif ".mul" in input.name or ".add" in input.name:
                            # Not handled right now
                            pass

        return graph.cleanup().toposort()