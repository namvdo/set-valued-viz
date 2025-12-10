from _imports import *

##from normals_model import ModelConfiguration as NormalsModel
##from mask_model import ModelConfiguration as MaskModel

def text_box(text,
             t="_", l="|", r="|", b="#",
             pad_t=1, pad_l=1, pad_b=1, pad_r=1,
             tl=None, tr=None, bl=None, br=None):
    
    lines = text.split("\n")#RE_NOTEMPTY_LINE.findall(text)
    text_width = len(max(lines, key=lambda x:len(x)))
    for i,line in enumerate(lines):
        lines[i] += " "*(text_width-len(line))
    lines = [" "*text_width]*pad_t+lines+[" "*text_width]*pad_b
    text_width += pad_l+pad_r+len(r)+len(l)

    text = "\n".join(lines)

    l = (l+" "*pad_l)
    r = (" "*pad_r+r)
    
    text = l+text.replace("\n", r+"\n"+l)+r
    text = (t*text_width+"\n" if t else "")+text+("\n"+b*text_width if b else "")

    if tl is not None: text = tl+text[1:]
    if tr is not None:
        text1, text2 = text.split("\n", 1)
        text = text1[:-1]+tr+"\n"+text2
    if bl is not None:
        text1, text2 = text.rsplit("\n", 1)
        text = text1+"\n"+bl+text2[1:]
    if br is not None: text = text[:-1]+br
    return text

def clean_text_box(text, **kwargs_override):
    kwargs = {
        "t": "─",
        "b": "─",
        "l": "│",
        "r": "│",
        "tl": "┌",
        "tr": "┐",
        "bl": "└",
        "br": "┘",
        }
    kwargs |= kwargs_override
    return text_box(text, **kwargs)




class ConsoleInterface():
    save_directory = ""
    interface_depth = 0
    
    def _depth_print(self, string, offset=0):
        depth = " "*(max(self.interface_depth+offset, 0)*4)
        string = string.replace("\n", "\n"+depth)
        print(depth+string)
        
    def _depth_input(self, string, *args, offset=0, **kwargs):
        depth = " "*(max(self.interface_depth+offset, 0)*4)
        string = string.replace("\n", "\n"+depth)
        return input(depth+string, *args, **kwargs)

    def _quit(self): return True
    
    def _show_user_options(self, desc, options):
        desc = clean_text_box(desc, pad_t=0, pad_b=0)
        
        string = ""
        for i,x in enumerate(options):
            if i!=0: string += "\n"
            string += str(i)+". "+x[0]
        
        string = clean_text_box(string, l="* ", pad_r=2, pad_t=0, pad_b=0)
        self._depth_print(desc+"\n"+string)
        
    def _process_user_input(self, options):
        user_input = self._depth_input("SELECT: ")
        if user_input and user_input.isnumeric():
            user_input = int(user_input)
            if 0<=user_input<len(options):
                f = options[user_input][1]
                if f is not None:
                    self._depth_print("-> "+options[user_input][0])
                    self.interface_depth += 1
                    output = f()
                    self.interface_depth -= 1
                    return output
                else:
                    self._depth_print("Not implemented")
            else:
                self._depth_print("Not a valid option")
        else: self._depth_print("Invalid input")
        print("")
        return False
        
    def _set_attr(self, attribute, negative_ok=False, zero_ok=False, target=None, description=None):
        if description is not None:
            self._depth_print(clean_text_box(description, pad_t=0, pad_b=0))
        
        if target is None: target = self
        current_value = getattr(target, attribute, None)
        t = type(current_value)
        
        user_input = self._depth_input(f"{current_value} -> ")
        success = False
        if user_input:
            if t in [int, float]:
                user_input = user_input.replace(",",".")
                if RE_INT.match(user_input) or (t is float and RE_FLOAT.match(user_input)):
                    user_input = t(user_input)
                    if user_input>0 or (negative_ok and user_input<0) or zero_ok:
                        setattr(target, attribute, user_input)
                        success = True
                if not success:
                    complain = "Value must be a "
                    if not zero_ok: complain += "non-zero "
                    if not negative_ok: complain += "positive "
                    complain += "integer" if t is int else "float"
                    self._depth_print(complain)
            elif t is str:
                setattr(target, attribute, user_input)
                success = True
            elif t is UserModifiablePoint:
                m = RE_2D_POINT.match(user_input)
                if m:
                    new_point = UserModifiablePoint(float(m.group(1)), float(m.group(2)))
                    setattr(target, attribute, new_point)
                    success = True
                else:
                    self._depth_print("Not a recognizable 2D point")
            else:
                self._depth_print(f"Value setting of type ({t}) is not defined")
        else: self._depth_print("Invalid input")
        if success:
            self._depth_print("Value set")
        print("")
        return success

    def _list_files(self, directory, ext):
        saves = []
        for f in list_files(directory):
            if f.endswith(f".{ext}"):
                name = os.path.basename(f).rsplit(".", 1)[0]
                size = readable_filesize(getsize(f))
                saves.append((name,size,f))
        
        if saves:
            max_name_len = len(max(saves, key=lambda x:len(x[0])))
            max_size_len = len(max(saves, key=lambda x:len(x[1])))
            
            desc = ""
            for name,size,f in saves:
                if desc: desc += "\n"
                desc += f"{name}"+"."*(10 + max_name_len*2-len(name) + max_size_len*2-len(size))+size
            
            desc = clean_text_box(desc, l=">", pad_t=0, pad_b=0)
        else:
            desc = "No valid files found"
        self._depth_print(desc)
        return bool(saves)

    def _save_file(self, name, ext, obj):
        save(os.path.join(self.save_directory, f"{name}.{ext}"), obj)
    
    def _select_file(self, ext):
        self._list_files(self.save_directory, ext)
        return self._depth_input("File name: ")

    def _load_file(self, name, ext):
        f = os.path.join(self.save_directory, f"{name}.{ext}")
        if os.path.isfile(f):
            obj = load(f)
            if obj is not None: return obj
            else: self._depth_print("Failed to load: '{f}'")
        else: self._depth_print(f"File not found: '{f}'")

    def start(self, menu_constructor_name):
        menu_constructor = getattr(self, menu_constructor_name, None)
        while menu_constructor is not None:
            desc, options = menu_constructor()
            self._show_user_options(desc, options)
            if self._process_user_input(options):
                break



class ConsoleInterface(ConsoleInterface):
    save_directory = os.path.join(WORKDIR, "saves")

    def deeper_examplemenu(self):
        self.start("examplemenu")
    
    def examplemenu(self) -> (str, list):
        desc = "An example description about the available options"
        options = [
            ("Deeper", self.deeper_examplemenu),
            ("Exit", self._quit),
            ]
        return desc, options


if __name__ == "__main__":
    ConsoleInterface().start("examplemenu")

