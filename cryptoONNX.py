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
        Prende un oggetto gs.Graph, cifra i pesi in-place e aggiunge gli attributi HMAC.
        """
        for node in graph.nodes:
            if len(node.inputs) > 1:
                for inp in node.inputs:
                    if isinstance(inp, gs.Constant):
                        
                        # Weights 
                        if ".weight" in inp.name:
                            shape_orig = inp.values.shape


                            weights_tensor = np.array(inp.values, dtype=np.int8).transpose(0, 2, 3, 1)
                            
                            encrypted_bytes = self._encrypt_weights(weights_tensor)
                            encrypted_bytes = encrypted_bytes.reshape(weights_tensor.shape)
                            
                            digest = hmac.new(self.hmac_key, np.ascontiguousarray(encrypted_bytes), hashlib.sha256).digest()
                            hmac_array = np.frombuffer(digest, dtype=np.uint32)
                            
                            inp.values = np.array(encrypted_bytes.transpose(0, 3, 1, 2), dtype=np.int8).reshape(shape_orig)
                            
                            node.attrs['hmac'] = gs.Constant(name=f"{node.name}_hmac", values=hmac_array)

                        # MatMul constants
                        elif "MatMul" in inp.name or (hasattr(inp, 'name') and "fc" in inp.name):
                            shape_orig = inp.values.shape
                            weights_tensor = np.array(inp.values, dtype=np.int8).transpose(1, 0)
                            
                            encrypted_bytes = self._encrypt_weights(weights_tensor)
                            encrypted_bytes.shape = weights_tensor.shape
                            
                            inp.values = np.array(encrypted_bytes.transpose(1, 0), dtype=np.int8).reshape(shape_orig)

                        # Biases
                        elif ".mul" in inp.name or ".add" in inp.name:
                            # Not handled right now
                            pass

        return graph.cleanup().toposort()