import sys, os
sys.path.insert(0, 'lib')
from comms_client_unencrypted import start_client, Client


TESTADDRESS = ("localhost", 12300)

def test():
    c = start_client(Client, TESTADDRESS, ".", "client_key")
    
    commands = [
        "uptime",
        "login asd 123",
        "logout",
        "register asd 123",
        "login asd 123",
        "login asd 123",
        "help",

        "non_existant_cmd",
        "read information",
        "write information 009900",
        "read information",
        
        "logout",
        "uptime",
        "help",
        ]
    
    for x in commands:
        if " " in x: print(x, "->", c(*x.split()))
        else: print(x, "->", c(x))
        print("")

if __name__ == "__main__":
##    test()
    
    c = start_client(Client, TESTADDRESS, ".", "client_key")
    while 1:
        x = input("input: ")
        if x:
            out = c(x)
            print(out)
    pass
