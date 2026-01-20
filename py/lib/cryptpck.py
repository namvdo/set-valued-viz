import os

from base64 import encodebytes as bs64enc
from base64 import decodebytes as bs64dec
from cryptography.fernet import Fernet as frnt
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.backends import default_backend
from cryptography.fernet import InvalidToken
from base64 import urlsafe_b64encode as urlsafeb64enc

import string
import secrets
import json

import numpy as np

# from primepck

def is_prime(x): # check for primes
    if type(x)==np.ndarray:
        x = np.int_(x)
        notvalid = np.logical_or(x<=1, np.logical_or(np.mod(x, 2)==0,np.mod(x, 3)==0))
        i = 5
        while not (i**2>x[~notvalid]).all():
            notvalid[~notvalid] = np.logical_or(x[~notvalid]%i==0, x[~notvalid]%(i+2)==0)
            i += 6
        return np.logical_or(np.logical_or(~notvalid, x==2), x==3)
    if x<=3: return x>1
    elif x%2==0 or x%3==0: return False
    i = 5
    while i**2<=x:
        if x%i==0 or x%(i+2)==0: return False
        i += 6
    return True

def find_primes(low:int, high:int=0, step:int=1, limit:int=0) -> int:
    low += (1-low%2)
    if 2>=low:
        yield 2
        low = 3
    i = 0
    count = 0
    while (high<1 or (low+i)<=high) and (limit<1 or limit>count):
        if is_prime(low+i):
            if not count%step: yield low+i
            count += 1
        i += 2

#



def new_password(length):
    alphabet = string.ascii_letters+string.digits#+"åöäÅÖÄ"
    while True:
        password = ''.join(secrets.choice(alphabet) for i in range(length))
        if length<5: return password # prevent inf. looping
        if (any(c.islower() for c in password)
            and any(c.isupper() for c in password)
            and sum(c.isdigit() for c in password)>=3): break
    return password


# cypher == generated in the script from a key
# key == either read from a separate file or generated from salt + password
# salt == binary data read from a separate file to match with password
##def bytespack(b): return str(bs64enc(b).hex())
##def bytesunpack(s): return bs64dec(bytes.fromhex(s))
##def encrypt(cypher, b) -> bytes: return bs64enc(cypher.encrypt(b))
##def decrypt(cypher, b) -> bytes: return cypher.decrypt(bs64dec(b))
##def packuplayer(cypher, string): return encrypt(cypher, string.encode("utf-8")).decode("utf-8")
##def unpacklayer(cypher, string): return decrypt(cypher, string.encode("utf-8")).decode("utf-8")

##def object_bytes(obj, cypher) -> bytes: return cypher.encrypt(json.dumps(obj).encode("utf8"))
##def bytes_object(b, cypher) -> object: return json.loads(cypher.decrypt(b).decode("utf8"))

# secrets
def randomint(i): return secrets.randbelow(i)+i//2
def randomstr(n): return secrets.token_hex(randomint(n))
def attachconfusion(string): return randomstr(8)+"_"+string+"_"+randomstr(8)
def removeconfusion(string): return string.rsplit("_", 1)[0].split("_", 1)[1]

def keyfrompassword(password, salt=None, urlsafe=True, salt_len=16):
    if salt is None: salt = os.urandom(salt_len) # need to save somewhere
    kdf = Scrypt(
        salt=salt,
        length=32,
        n=2**14,
        r=8,
        p=1,
        backend=default_backend(),
        )
    key = kdf.derive(password.encode())
    if urlsafe: return urlsafeb64enc(key), salt
    return key, salt
def verifypassword(password, key, salt):
    kdf = Scrypt(
        salt=salt,
        length=32,
        n=2**14,
        r=8,
        p=1,
        backend=default_backend(),
        )
    kdf.verify(password.encode(), key)
    return True





def RSA(p:int, q:int, e=65537):
    n = p*q
    public = (n, e) # sent to the other party
    euler_pq = (p-1)*(q-1)
    for s in range(euler_pq): # range(euler_pq-1, 0, -1)
        if (e*s)%euler_pq==1:
            break # secret found
    return (n, s), public

def RSA_process(x:int, key):
    mod, e = key
    result = 1
    i = 1
    while e!=0:
        if e%2:
            result *= (x**i)%mod
            result %= mod
        e >>= 1
        i <<= 1
    return result

def check_signature(x:int, signature:int, key):
    return signature==RSA_process(x, key)

class RSA_connection():
    secret = None
    public = None
    def __init__(self):
        pg1 = find_primes(randomint(100), high=0, step=randomint(20))
        pg2 = find_primes(randomint(100)*((next(pg1)//3)%10+2), high=0, step=randomint(50))
        pg3 = find_primes(randomint(1000)*(((next(pg2)//3)%10+2)*5), high=0, step=randomint(5))
        n, s, e = next(pg1), next(pg2), 65537 # 683, 1187, 1699
        self.secret, self.public = RSA(n, s, e)

def bytes_RSA_json(x:bytes, key):
    return json.dumps([RSA_process(i, key) for i in x], separators=(',', ':'))
def json_RSA_bytes(x:str, key):
    return bytes([RSA_process(i, key) for i in json.loads(x)])



if __name__ == "__main__":
##    from cryptography.hazmat.primitives import ciphers
##    
##    key = os.urandom(32)
##    print(key)
##    iv = os.urandom(16)
##    c = ciphers.Cipher(ciphers.algorithms.AES(key), ciphers.modes.CBC(iv))
##
##    original = b"message"
##    print(original, len(original))
##    original += b"0"*(16-(len(original)%16))
##    
##    encryptor = c.encryptor()
##    encrypted = encryptor.update(original)+encryptor.finalize()
##    print(encrypted)
##    decryptor = c.decryptor()
##    decrypted = decryptor.update(encrypted)+decryptor.finalize()
##    print(decrypted)

    
##    p = new_password(4)
##    print(p)
    
##    c1 = RSA_connection()
##    print(c1.secret)
##    print(c1.public)
##    msg = b"asd123645"
##    print(msg)
##    s = bytes_RSA_json(msg, c1.public)
##    print(json_RSA_bytes(s, c1.secret))
    
####    c2 = RSA_connection()
####    c1.targets["2"] = c2.public
####    c2.targets["1"] = c1.public
####    print("1", c1.secret, "-", c1.public, c1.targets)
####    print("2", c2.secret, "-", c2.public, c2.targets)
####    key = frnt.generate_key()
####    
####    print(key)
####    key_encr = bytes_RSA_json(key, c2.public)
####    print("1 sends to 2")
####    print(key_encr)
####    key_decr = json_RSA_bytes(key_encr, c2.secret)
####    print("2 receives from 1")
####    print(key_decr)
####    
####    cypher = frnt(key_decr)

    
    
##    for i in range(8): print(randomstr(i+3))
    
##    p = "password\t\n$".encode("utf8")
##    p = p.decode("ascii")
##    h = pbkdf2_sha256.hash(p)
##    print(p, h)
    
##    key, salt = keyfrompassword("asd")
##    cypher = frnt(key)
##    t = "abcdefg"
##    print(t, decrypt(cypher, encrypt(cypher, t)))
    
##    with open(".\\test.txt", "w") as f:
##        f.write("hello")
##    fileencrypt(".\\test.txt", "asd")
##    filedecrypt(".\\test.txt", "asd")
    pass

