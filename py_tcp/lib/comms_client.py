from comms import *



class Client():
    def __init__(self, address, key, buffer=4096):
        self.address = address
        self.default_cypher = cryptpck.frnt(key) # load_key(os.path.join(path, "version"))
        self.buffer = buffer
        self.latency = 0
        self.reset()
    
    def encrypt(self, *args, **kwargs):
        rsa_keys = cryptpck.RSA_connection() # onetime
        key = self(f"encrypt", rsa_keys.public, **kwargs, local_ok=False) # request key from server ->
        # <- server initializes cypher and sends a RSA encrypted cypher key
        if key:
            key = cryptpck.json_RSA_bytes(key, rsa_keys.secret)
            self.cypher = cryptpck.frnt(key)
            return True
        return False
    def reset(self, *args, **kwargs):
        self.identifier = cryptpck.new_password(IDENTIFIER_LEN) # random instance identifier
        self.cypher = self.default_cypher
        return True

##    @timepck.func_timer_decor
    def __call__(self, k, *args, recv=True, timeout=5, local_ok=True, **kwargs):
        f_args, f_kwargs = read_args_kwargs(k)
        k = str(f_args[0])
        args = (*f_args[1:], *args)
        kwargs = f_kwargs|kwargs
        x = getattr(self, k, None)
        if local_ok and x and x!=self.__call__ and isinstance(x, type(self.__call__)): return x(*args, **kwargs) # local macros
        if not k[0].isdigit():
            obj = (k, args, kwargs)
            b_in = bytes(self.identifier, "ascii")+self.cypher.encrypt(object_to_bytes(obj))
            b_out = b""
            
            t_start = timepck.nspec()
            for b in quicksend(self.address, b_in, recv=recv, timeout=timeout, buffer=self.buffer): b_out += b
            self.latency = (timepck.nspec()-t_start)/1e6
            
            if b_out: return bytes_to_object(self.cypher.decrypt(b_out))


def start_client(client_class, address, local_dir, key_name):
    key = load_key(f"{local_dir}/{key_name}")
    return client_class(address, key)



if __name__ == "__main__":
    address = ("localhost", 48213)
    cc = start_client(Client, address, "./comms_client", "version")
    while 1:
        out = cc(input("in: "))
        print(out, end="\n\n")
