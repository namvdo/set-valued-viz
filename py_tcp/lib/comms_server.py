from comms import *


class Server(socketserver.TCPServer):
    PRINT = False
    
    # only functions with 'instance' keyword argument are executable by clients
    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            b_in = self.request.recv(self.server.buffer)
            ip = self.client_address[0]
            
            identifier = b_in[:IDENTIFIER_LEN].decode('ascii')
            instance = f"{ip}+{identifier}"
            cypher = self.server.cyphers.get(instance, self.server.default_cypher)
            k, args, kwargs = bytes_to_object(cypher.decrypt(b_in[IDENTIFIER_LEN:]))
            
            if args is None: args = ()
            if kwargs is None: kwargs = {}
            
            if k in self.server.client_functions:
                kwargs["instance"] = instance
                if func:=getattr(self.server, k, None):
                    if self.server.PRINT: print(k, args, kwargs)
                    result = func(*args, **kwargs)
                    if self.server.PRINT: print("->", result)
                    b_out = cypher.encrypt(object_to_bytes(result))
                    self.request.sendall(b_out)

    def __init__(self, address, key, path, buffer=4096):
        super().__init__(address, self.Handler)
        self.timenow = timepck.now()
        self.path = path
        self.buffer = buffer
        self.default_cypher = cryptpck.frnt(key)
        
        self.cyphers = {} # instance -> cypher
        
        self.stopwatch = timepck.Stopwatch()
        self.timer = FuncTimer()
        self.server_init()
        self.__build_function_help()
        
    def service_actions(self):
        self.stopwatch()
        self.timer(self.stopwatch.lap/1e6)
        self.timenow = timepck.now()
    def server_close(self):
        while len(self.timer): self.timer(1000) # force trigger ongoing timed functions
        
    def __build_function_help(self):
        self.client_functions = {}
        for k in dir(self):
            if (f:=getattr(self, k, None)) is not None and f!=self.__init__ and isinstance(f, type(self.__init__)):
                sig_str = str(inspect.signature(f))
                i_end = len(sig_str)-(sig_str[::-1].index(")"))-1
                output_str = sig_str[i_end+2:]
                args, kwargs = read_args_kwargs(sig_str[1:i_end])
                if "instance" in kwargs:
                    del kwargs["instance"]
                    help_string = ""
                    if args: help_string += " ".join(f"<{x}>"for x in args)
                    if kwargs: help_string += " "*bool(args)+" ".join(f"<{k}={v}>" for k,v in kwargs.items())
                    d = self.client_functions
                    d[k] = help_string+" "*bool(help_string)+(output_str if output_str else "-> ?")

    def help(self, function_name=None, instance=None):
        # return available args and kwargs for a function
        if function_name in self.client_functions: return function_name+" "+self.client_functions.get(function_name)
        elif function_name:
            options = list(self.client_functions.keys())
            options = strsearch(function_name, options, amount=len(options))
            if options: return self.help(options[0], instance)
        return "\nHELP:"+"\n| ".join([""]+list(self.client_functions.keys()))
        
    def uptime(self, instance=None) -> int: return int(self.stopwatch.total/1e6)
    def datetime(self, instance=None) -> str: return self.timenow.isoformat()

    def encrypt(self, RSA_public_key, hours=0, minutes=0, seconds=0, ms=0, instance=None) -> str:
        x = RSA_public_key
        s = max(ms//1000+(hours*3600+minutes*60+seconds), 0)
        if s==0: s = 300 # 5 min default
        if not is_str(x) and is_iter(x) and len(x)==2:
            key = cryptpck.frnt.generate_key()
            self.cyphers[instance] = cryptpck.frnt(key)
            timer_name = instance+"_encrypt_expire"
            if self.timer.exists(timer_name):
                self.timer.replace(timer_name, s*1000)
            else:
                self.timer.new(timer_name, s*1000)
                self.timer.set(timer_name, self._encrypt_expire, instance)
            return cryptpck.bytes_RSA_json(key, x)
        return ""
    def encrypt_expires(self, instance=None) -> int: # seconds
        return int(self.timer.get(instance+"_encrypt_expire", 0))//1000
    def _encrypt_expire(self, inst):
        if inst in self.cyphers: del self.cyphers[inst]
        

class Server(Server):
    def server_init(self):
        pass
##    def service_actions(self):
##        super().service_actions()
##    def server_close(self):
##        super().server_close()

    def hello(self, instance=None) -> int:
        return 123



def start_server(server_class, address, local_dir, key_name, buffer=4096, pollrate=0.2):
    key = load_key(f"{local_dir}/{key_name}")
    with server_class(address, key, local_dir, buffer) as s:
        s.serve_forever(pollrate)
        s.server_close()

