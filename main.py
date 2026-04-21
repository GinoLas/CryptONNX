import onnx
import onnx_graphsurgeon as gs
import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import os

graph = gs.import_onnx(onnx.load("files/network_og.onnx"))

key = bytes.fromhex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF")  
nonce = bytes.fromhex("000000000000000000000000CACACACA") 
backend = default_backend()

def encrypt_weights(data_numpy):
    raw_bytes = data_numpy.tobytes()
    
    cipher = Cipher(algorithms.AES(key), modes.CTR(nonce), backend=backend)
    encryptor = cipher.encryptor()
    encrypted_data = encryptor.update(raw_bytes) + encryptor.finalize()
    
    return np.frombuffer(encrypted_data, dtype=np.uint8)


for node in graph.nodes:
        # print("-------" + node.name + "-------")
        # print(node.attrs)
        if(len(node.inputs) > 1):
            for input in node.inputs :
                if isinstance(input, gs.Constant):
                    print(f"Cifratura del buffer: {input.name}")

                    if(".weight" in input.name):
                        print(f"{input.name} --> peso")
                        shape = input.values.shape
                        weights_tensor = np.array(input.values,dtype = np.int8)
                        weights_tensor = weights_tensor.transpose(0,2,3,1)
                        for i, byte in enumerate(weights_tensor.flatten()):
                            print(str(byte),end="\t")
                        
                            if (i + 1) % 8 == 0:
                                print(",")  
                            else:
                                print(", ",end="") 
                        encrypted_bytes = encrypt_weights(weights_tensor)
                        encrypted_bytes.shape = weights_tensor.shape
                        input.values = np.array(encrypted_bytes.transpose(0,3,1,2),dtype = np.int8)
                        input.values.shape = shape

                    elif(".mul" in input.name or ".add" in input.name):
                        print("Beh... non ho idea di cosa si possa fare...")


            
                    elif("MatMul" in input.name):
                        print(f"{input.name} --> mat mul constant")
                        shape = input.values.shape
                        weights_tensor = np.array(input.values,dtype = np.int8)
                        weights_tensor = weights_tensor.transpose(1,0)
                        for i, byte in enumerate(weights_tensor.flatten()):
                            print(str(byte),end="\t")
                        
                            if (i + 1) % 8 == 0:
                                print(",")  
                            else:
                                print(", ",end="") 
                        encrypted_bytes = encrypt_weights(weights_tensor)
                        encrypted_bytes.shape = weights_tensor.shape
                        input.values = np.array(encrypted_bytes.transpose(1,0),dtype = np.int8)
                        input.values.shape = shape

                
graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "files/network.onnx")
