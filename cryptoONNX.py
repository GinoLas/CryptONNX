import onnx
import onnx_graphsurgeon as gs
import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import hmac
import hashlib
import secrets
import base64
 
class ONNXEncryptor:
    def __init__(self, key_hex: str, hmac_key_hex: str):
        self.key = bytes.fromhex(key_hex)
        self.hmac_key = bytes.fromhex(hmac_key_hex)
        self.key_hex = key_hex
        self.hmac_key_hex = hmac_key_hex
        self.backend = default_backend()
 
    def _ctr_encrypt(self, data_numpy: np.ndarray, iv) -> bytes:
        """AES-CTR di un tensore, senza padding: CTR cifra anche un ultimo blocco parziale
        (usa solo i primi byte del keystream). Se la dimensione non è multipla di 16 byte
        stampa le informazioni sul tensore."""
        raw_bytes = data_numpy.tobytes()
        n = len(raw_bytes)

        cipher = Cipher(algorithms.AES(self.key), modes.CTR(iv), backend=self.backend)
        encryptor = cipher.encryptor()
        encrypted_data = encryptor.update(raw_bytes) + encryptor.finalize()

        if n % 16 != 0:
            short = lambda b: b.hex() if len(b) <= 64 else b[:32].hex() + " ... " + b[-32:].hex()
            print("[CTR] ---- tensore non multiplo di 16 byte ----")
            print(f"[CTR] dtype={data_numpy.dtype}, shape={data_numpy.shape}, elementi={data_numpy.size}")
            print(f"[CTR] size = {n} byte = {n // 16} blocchi interi + {n % 16} byte (servirebbero {(-n) % 16} di padding)")
            print(f"[CTR] iv  = {bytes(iv).hex()}")
            print(f"[CTR] key = {self.key[:2].hex()}...{self.key[-2:].hex()}")
            print("[CTR] vettore originale:", np.array2string(data_numpy.reshape(-1), threshold=64, edgeitems=8))
            print("[CTR] originale (hex):  ", short(raw_bytes))
            print("[CTR] cifrato (hex):    ", short(encrypted_data))

        return encrypted_data

    def _ctr_bytes(self, raw: bytes, iv) -> bytes:
        """AES-CTR su byte grezzi: cifrare e decifrare sono la stessa operazione."""
        cipher = Cipher(algorithms.AES(self.key), modes.CTR(iv), backend=self.backend)
        encryptor = cipher.encryptor()
        return encryptor.update(raw) + encryptor.finalize()

    def broadcast_encrypted(self, encrypted: np.ndarray, iv, new_shape) -> np.ndarray:
        """Costante già cifrata da espandere a `new_shape`: decifra, espande il chiaro e ricifra
        il vettore intero con lo stesso IV, come un unico flusso CTR. Restituisce un array con lo
        stesso dtype di `encrypted` e forma `new_shape`."""
        dtype = encrypted.dtype
        plain = np.frombuffer(self._ctr_bytes(encrypted.tobytes(), iv), dtype=dtype).reshape(encrypted.shape)
        expanded = np.ascontiguousarray(np.broadcast_to(plain, new_shape))
        new_cipher = self._ctr_bytes(expanded.tobytes(), iv)
        return np.frombuffer(new_cipher, dtype=dtype).reshape(new_shape).copy()

    def _encrypt_weights(self, data_numpy: np.ndarray, iv) -> np.ndarray:
        return np.frombuffer(self._ctr_encrypt(data_numpy, iv), dtype=np.uint8)

    def _encrypt_biases(self, data_numpy: np.ndarray, iv) -> np.ndarray:
        return np.frombuffer(self._ctr_encrypt(data_numpy, iv), dtype=np.uint32)
 
    def process_graph(self, graph: gs.Graph) -> gs.Graph:
        """
        Prende un oggetto gs.Graph, cifra i pesi e aggiunge come attributo ai nodi HMAAC sui pesi.
        """
 
        print(f"hmac_key = {self.hmac_key}")
        print(f"Key = {self.key}")
 
        #Aggiunta delle due chiavi al grafo come attributo python
 
        print("Adding keys...")
 
        graph.hmac_key = self.hmac_key_hex
        graph.aes_key = self.key_hex
 
        for node in graph.nodes:
            if len(node.inputs) > 1:
                for input in node.inputs:
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
                            iv = secrets.token_bytes(16)
                            print(iv.hex())
                            encrypted_bytes = self._encrypt_weights(weights_tensor,iv)
                            encrypted_bytes = encrypted_bytes.reshape(weights_tensor.shape)
                            digest = hmac.new(self.hmac_key, np.ascontiguousarray(encrypted_bytes), hashlib.sha256).digest()
                            hmac_array = np.frombuffer(digest,dtype=np.uint32)
                            # input.values = np.array(encrypted_bytes.transpose(0,3,1,2),dtype = np.int8)
                            input.values = np.array(encrypted_bytes,dtype = np.int8)
                            input.values = input.values.reshape(shape)
                            if('hmac' not in node.attrs):
                                node.attrs['hmac'] = True
                            node.attrs[f"{input.name}_hmac"] = (gs.Constant(
                                name = f"{input.name}_hmac",
                                values = hmac_array
                            ))
                            node.attrs[f"{input.name}_iv"] = (gs.Constant(
                                name = f"{input.name}_iv",
                                values = np.frombuffer(iv,dtype=np.uint8)
                            ))
 
                        # MatMul constants
                        elif "MatMul" in input.name or (hasattr(input, 'name') and "fc" in input.name):
                            shape = input.values.shape
                            weights_tensor = np.array(input.values, dtype=np.int8)
                            iv = secrets.token_bytes(16)
                            print(iv.hex())
                            encrypted_bytes = self._encrypt_weights(weights_tensor,iv)
                            encrypted_bytes.shape = weights_tensor.shape
                            digest = hmac.new(self.hmac_key, np.ascontiguousarray(encrypted_bytes), hashlib.sha256).digest()
                            hmac_array = np.frombuffer(digest,dtype=np.uint32)
                            input.values = np.array(encrypted_bytes,dtype = np.int32)
                            input.values = input.values.reshape(shape)
                            if('hmac' not in node.attrs):
                                node.attrs['hmac'] = True
                            node.attrs[f"{input.name}_hmac"] = (gs.Constant(
                                name = f"{input.name}_hmac",
                                values = hmac_array
                            ))
                            node.attrs[f"{input.name}_iv"] = (gs.Constant(
                                name = f"{input.name}_iv",
                                values = np.frombuffer(iv,dtype=np.uint8)
                            ))
                           
                            input.values = np.array(encrypted_bytes, dtype=np.int8)
 
                        # Biases
                        elif "mul_tensor" in input.name or "add_tensor" in input.name or "bias_tensor" in input.name:
                            shape = input.values.shape
                            weights_tensor = np.array(input.values,dtype = np.int32)
                            iv = secrets.token_bytes(16);
                            print(iv.hex())
                            encrypted_bytes = self._encrypt_biases(weights_tensor,iv)
                            encrypted_bytes = encrypted_bytes.reshape(weights_tensor.shape)
                            digest = hmac.new(self.hmac_key, np.ascontiguousarray(encrypted_bytes), hashlib.sha256).digest()
                            hmac_array = np.frombuffer(digest,dtype=np.uint32)
                            # input.values = np.array(encrypted_bytes.transpose(0,3,1,2),dtype = np.int8)
                            input.values = np.array(encrypted_bytes,dtype = np.int32)
                            input.values = input.values.reshape(shape)
                            if('hmac' not in node.attrs):
                                node.attrs['hmac'] = True
                            node.attrs[f"{input.name}_hmac"] = (gs.Constant(
                                name = f"{input.name}_hmac",
                                values = hmac_array
                            ))
                            node.attrs[f"{input.name}_iv"] = (gs.Constant(
                                name = f"{input.name}_iv",
                                values = np.frombuffer(iv,dtype=np.uint8)
                            ))
 
 
        return graph.cleanup().toposort()