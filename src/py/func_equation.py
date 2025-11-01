import re
import numpy as np


def find_parenthesis_start(string, start_index):
    index = 0
    closure = 0
    for i in range(start_index):
        c = string[start_index-i]
        match c:
            case "(": closure += 1
            case ")": closure -= 1
        if closure>=0:
            index = start_index-i
            break
    return index
def find_parenthesis_end(string, start_index):
    index = len(string)-1
    closure = -int(string[start_index]=="(")
    for i in range(len(string)-start_index):
        c = string[start_index+i]
        match c:
            case "(": closure += 1
            case ")": closure -= 1
        if closure<0:
            index = start_index+i
            break
    return index


RE_INT = r"(?:\-?\d+)"
RE_FLOAT = r"(?:(?:\-?\d*\.\d+)|(?:\-?\d+\.\d*))"
RE_NUMBER = rf"(?:{RE_FLOAT}|{RE_INT})"

RE_INT_POS = r"(?:\d+)"
RE_FLOAT_POS = r"(?:(?:\d*\.\d+)|(?:\d+\.\d*))"
RE_NUMBER_POS = rf"(?:{RE_FLOAT_POS}|{RE_INT_POS})"

RE_FACT = r"(?:(\d+)\!+)"
RE_FUNC = r"(?:a?(?:sin|cos|tan)|abs|log|ln)"
RE_ALGEBRAIC = r"(?:[a-zA-Z]+)"
RE_SUBSTITUTION = r"(?:\{\d+\})"
RE_ALGSUB = rf"(?:{RE_ALGEBRAIC}|{RE_SUBSTITUTION})"

RE_OPSUBS = r"(?:\{\<(\d+)\>\})"
RE_OPSUBS_NONCAP = r"(?:\{\<\d+\>\})"

RE_FUNC_PARENTHESIS = rf"({RE_FUNC})?\(([^\(\)]+)\)"
RE_FUNC_PARENTHESIS_NONCAP = rf"(?:{RE_FUNC})?\([^\(\)]+\)"
RE_NUM_ALGSUB = rf"(?:({RE_NUMBER})|({RE_ALGSUB}))"
RE_NUM_ALGSUB_NONCAP = rf"(?:(?:{RE_NUMBER})|(?:{RE_ALGSUB}))"

RE_OPS = r"[\*/\+\-]+"

##RE_CLUSTER = 

RE_USELESS_PARENTHESIS = rf"(({RE_FUNC}|{RE_OPS})?\(({RE_NUM_ALGSUB_NONCAP})\))" # detect useless parenthesis

algebraic_blacklist = {"sin","cos","tan","asin","acos","atan", "abs", "log", "ln"} # invalid variable names in equations

def split_equation(equation):
    index = 0
    temps = [""]
    for part in re.split(rf"({RE_OPS}|(?:(?:{RE_FUNC})?\()|\))", equation):
        if "(" in part: temps.append(part)
        elif ")" in part:
            temp = temps.pop()+part
            temps[-1] += "{<"+str(index)+">}"
            yield temp
            index += 1
        else: temps[-1] += part
    for temp in temps: yield temp


def order_of_operations(equation):
    equation = expand(equation)
    valid_target = rf"(?:{RE_NUM_ALGSUB_NONCAP}|"+r"(?:\{\<\d+\>\})|"+rf"(?:{RE_FUNC_PARENTHESIS_NONCAP}))"
    negation_pattern = rf"((?:\-){valid_target})"
    power_pattern = rf"({valid_target}(?:\*\*|\^){valid_target})"
    div_pattern = rf"({valid_target}/{valid_target})"
    prod_pattern = rf"({valid_target}\*{valid_target})"
    add_pattern = rf"({valid_target}\+{valid_target})"
    
    index = 0
    temps = [""]
    for part in re.split(rf"({RE_OPS}|(?:(?:{RE_FUNC})?\()|\))", equation):
        if "(" in part: temps.append(part)
        elif ")" in part:
            temp = temps.pop()+part
            
            for pattern in [negation_pattern,power_pattern,div_pattern,prod_pattern,add_pattern]:
                while (found:=re.findall(pattern, temp)):
                    for op in found:
                        sub = "{<"+str(index)+">}"
                        index += 1
                        temp = temp.replace(op, sub, 1)
                        yield op
            
            temps[-1] += "{<"+str(index)+">}"
            index += 1
            yield temp
            
        else: temps[-1] += part
        
    for temp in temps:
        for pattern in [negation_pattern,power_pattern,div_pattern,prod_pattern,add_pattern]:
            while (found:=re.findall(pattern, temp)):
                for op in found:
                    sub = "{<"+str(index)+">}"
                    index += 1
                    temp = temp.replace(op, sub, 1)
                    yield op
                
        yield temp





RE_EQJOINABLE = rf"(?:.*|^)({RE_OPSUBS})(?:.*|$)"
def join_equation(equation, parts):
##    print("JOINING", equation)
    while m:=re.match(RE_EQJOINABLE, equation):
        equation = equation.replace(m.group(1), parts[int(m.group(2))])
##    print("JOINED", equation)
    return equation



def get_full_parenthesis_from_start(string, substring):
    start_i = string.index(substring)+substring.index("(")
    end_i = find_parenthesis_end(string, start_i)+1
    return string[start_i:end_i]
def get_full_parenthesis_from_end(string, substring):
    start_i = string.index(substring)
    end_i = find_parenthesis_start(string, start_i)+1
    return string[start_i:end_i]


def iterate_top_clusters(equation):
    ops = "+-"
    prev = ""
    temp = ""
    
    ii = 0
    for i,c in enumerate(equation):
        if ii>0:
            ii -= 1
            continue
        if c in ops:
            if temp and temp not in ops:
                if prev:
                    if temp.startswith("**"):
                        yield prev+temp
                        temp = ""
                    else: yield prev
                prev = temp
                temp = ""
            temp += c
        elif c=="(":
            ii = find_parenthesis_end(equation, i)
            temp += equation[i:ii+1]
            ii -= i
        else: temp += c
        
    if prev:
        if temp.startswith("**"):
            yield prev+temp
            temp = ""
        else: yield prev
    yield temp


def shrink(eq):
    pattern1 = rf"((?:\(|\+|^|{RE_ALGSUB})(\-?(?:{RE_NUMBER_POS}[\+\-]+)+(?:{RE_NUMBER_POS}))(?:\)|$|{RE_ALGSUB}))" # trivial addition operations
    pattern2 = rf"((?:0(?:\*+|/)(?:{RE_NUM_ALGSUB_NONCAP}|\())|(?:(?:{RE_NUM_ALGSUB_NONCAP}|\))\*0))" # zero multipliers
    pattern3 = rf"((?:{RE_NUM_ALGSUB_NONCAP}|\))\*\*0)" # zero power
    pattern4 = rf"((?:\(|^|([^\*][\*\+\-]))((?:{RE_NUMBER}\*)+(?:{RE_NUMBER}))(?:\)|$|[/\+\-]|{RE_ALGSUB}))" # trivial product operations
    pattern5 = rf"((?:\(|^|([\*\+\-/]))((?:{RE_NUMBER}\*\*)+(?:{RE_NUMBER}))(?:\)|$|[/\+\-]|[^(?:\*\*)]|{RE_ALGSUB}))" # trivial power operations

    eq = re.sub(r"\+\-|\-\+", "-", eq)
    done = False
    while not done:
        results = re.findall(RE_USELESS_PARENTHESIS, eq)
        rejected = 0
        for whole,prefix,inside in results:
            if prefix not in algebraic_blacklist:
                eq = eq.replace(whole, prefix+inside, 1)
            else:
                rejected += 1
            
        results1 = re.findall(pattern1, eq)
        for whole,inside in results1:
            total = 0
            negative = 0
            for x in re.split(r"(\+|\-)", inside):
                if x=="-": negative += 1
                elif x=="+": negative = 0
                elif x:
                    if negative%2: total -= float(x)
                    else: total += float(x)
            if int(total)==total: total = int(total)
            
            i = eq.index(whole)
            eq = eq[:i]+eq[i:].replace(inside, "+"*int(total>=0 and inside[0]=="-")+str(total), 1)
        
        results2 = re.findall(pattern2, eq)
##        print(results2)
        for target in results2:
            if target[0]==")":
                end_i = eq.index(target)
                start_i = find_parenthesis_start(eq, end_i)
                end_i += len(target)
##                print("zeroing", eq[start_i:end_i+1])
                start_i -= len(re.match(r"(?:.*[^a-zA-Z]|^)([a-zA-Z]*)$", eq[:start_i])[1]) # remove prefix len aswell
                eq = eq[:start_i]+"0"+eq[end_i:]
            elif target[-1]=="(":
                start_i = eq.index(target)
                end_i = find_parenthesis_end(eq, start_i+len(target))
##                print("zeroing", eq[start_i:end_i+1])
                eq = eq[:start_i]+"0"+eq[end_i+1:]
            else:
##                print("zeroing", target)
                eq = eq.replace(target, "0")
        
        results3 = re.findall(pattern3, eq)
        for target in re.findall(pattern3, eq):
            if target[0]==")":
                end_i = eq.index(target)
                start_i = find_parenthesis_start(eq, end_i)
                end_i += len(target)
##                print("oneing", eq[start_i:end_i])
                start_i -= len(re.match(r"(?:.*[^a-zA-Z]|^)([a-zA-Z]*)$", eq[:start_i])[1]) # remove prefix len aswell
                eq = eq[:start_i]+"1"+eq[end_i:]
            else:
                eq = eq.replace(target, "1")

        results5 = re.findall(pattern5, eq)
        for whole,separating_op,inside in results5:
            total = None
            for x in re.split(r"\*\*", inside):
                if total is None: total = float(x)
                elif x: total **= float(x)
            if int(total)==total: total = int(total)
            i = eq.index(whole)+len(separating_op)
            eq = eq[:i]+eq[i:].replace(inside, str(total), 1)
        
        results4 = re.findall(pattern4, eq)
        for whole,separating_op,inside in results4:
            total = 1.
            for x in re.split(r"\*", inside):
                if x: total *= float(x)
            if int(total)==total: total = int(total)
            i = eq.index(whole)+len(separating_op)
            eq = eq[:i]+eq[i:].replace(inside, str(total), 1)
        
        if len(results)==rejected and not results1 and not results2 and not results3 and not results4 and not results5:
            done = True
    
    # remove useless ones  # "**1" & "/1" -> ""
    for target,end in re.findall(rf"([\*/^]+1)([\*/\)]|$|\D)", eq):
        eq = eq.replace(target, "")
##    for start,target in re.findall(rf"(^|[\*/\(]|\D)(1[\*]+)", eq):
##        eq = eq.replace(target, "")

    # remove useless zeros
    for whole,target,end in re.findall(rf"(([\+\-]+0+)([\+\-\)]|$))", eq):
        i = eq.index(whole)
        eq = eq[:i]+eq[i:].replace(target, "", 1)
    for whole,start,target in re.findall(rf"(([\+\-\(]|^)(0+[\+\-]+))", eq):
        i = eq.index(whole)
        eq = eq[:i]+eq[i:].replace(target, "", 1)
        
    # remove useless product symbols
    pattern = rf"(({RE_NUMBER}|[\(\)])\*({RE_ALGSUB}|[\(\)]))"
    for whole,x,y in re.findall(pattern, eq): eq = eq.replace(whole, f"{x}{y}")
    pattern = rf"(({RE_NUMBER})\*([\(\)]))"
    for whole,x,y in re.findall(pattern, eq): eq = eq.replace(whole, f"{x}{y}")

    # shrink power symbols
    pattern = rf"(({RE_NUM_ALGSUB_NONCAP}|\))\*\*({RE_NUM_ALGSUB_NONCAP}|\())"
    for whole,x,y in re.findall(pattern, eq): eq = eq.replace(whole, f"{x}^{y}")

    # double parenthesis
    while m:=re.match(r".*(\(\(([^\)\(]*)\)\)).*", eq):
        eq = eq.replace(m.group(1), f"({m.group(2)})")
    return eq




def expand(eq):
    # make subtraction symbol "+-"; then negation symbol is "-"
    pattern = rf"(({RE_NUM_ALGSUB_NONCAP})(?:\-({RE_NUM_ALGSUB_NONCAP}|\()))"
    for whole,x,y in re.findall(pattern, eq):
        if y=="(":
            i = eq.index(whole)+len(x)+1
            y = eq[i:find_parenthesis_end(eq, i)+1]
            whole = whole[:-1]+y
        eq = eq.replace(whole, f"{x}+-{y}", 1)
    
    # restore default power symbols
    pattern = rf"(({RE_NUM_ALGSUB_NONCAP}|\))\^({RE_NUM_ALGSUB_NONCAP}|\())"
    for whole,x,y in re.findall(pattern, eq): eq = eq.replace(whole, f"{x}**{y}")
    
    # restore product symbol
    pattern = rf"((?:({RE_FUNC})|({RE_NUMBER}|\)))({RE_FUNC}|{RE_ALGSUB}|\())"
    for whole,f,x,y in re.findall(pattern, eq):
        if f: continue
        i = eq.index(whole)+len(x)
        eq = eq[:i]+"*"+eq[i:]
    return eq


def beautify(eq):
    eq = expand(eq)
    splits = list(split_equation(eq))
    splits = [shrink(split) for split in splits]
    eq = join_equation(splits[-1], splits)
    return shrink(eq)



def _advanced_solve_setup(equation, **input_values):
    i = 0
    values = {}
    substitutions = {}
    string_index = 0
    for k in re.findall(RE_ALGEBRAIC, equation[string_index:]): # [^{RE_ALGEBRAIC}]*(
        if k not in algebraic_blacklist:
            a = "{"+str(i)+"}"
            substitutions[a] = k
            if k not in input_values: return None, None, None # not solvable, missing inputs
            string_index1 = equation[string_index:].index(k)
            equation = equation[:string_index]+equation[string_index:].replace(k, a, 1)
            string_index += string_index1
            i += 1
        string_index += len(k)
    return equation, values, substitutions

def _advanced_get_value(x:str, input_values:dict, values:dict, substitutions:dict):
    if x not in values:
        k = substitutions[x]
        if hasattr(input_values[k], "copy"): values[x] = input_values[k].copy()
        else: values[x] = input_values[k]
    return values[x]

def _advanced_operation_negation(equation:str, input_values:dict, values:dict, substitutions:dict):
    while l:=re.findall(rf"(\-({RE_ALGSUB}))", equation): # negate (-x)
        for whole,alg0 in l:
            x = _advanced_get_value(alg0, input_values, values, substitutions)
            values[alg0] = -x
            equation = equation.replace(whole, alg0, 1)
    return equation

def _advanced_operation_power(equation:str, input_values:dict, values:dict, substitutions:dict):
    while l:=re.findall(rf"({RE_NUM_ALGSUB}\*\*{RE_NUM_ALGSUB})", equation): # exp (x**y)
        for whole,num0,alg0,num1,alg1 in l:
            if alg0: x = _advanced_get_value(alg0, input_values, values, substitutions)
            else: x = float(num0)
            if alg1: z = _advanced_get_value(alg1, input_values, values, substitutions)
            else: z = float(num1)
            x = x**z
            if alg0 and alg1: # combined
                values[alg0] = x
                del values[alg1]
                equation = equation.replace(whole, alg0, 1)
            elif alg0:
                values[alg0] = x
                equation = equation.replace(whole, alg0, 1)
            elif alg1:
                values[alg1] = x
                equation = equation.replace(whole, alg1, 1)
            else: equation = equation.replace(whole, str(x), 1)
    return equation

def _advanced_operation_multdiv(equation:str, input_values:dict, values:dict, substitutions:dict):
    while l:=re.findall(rf"({RE_NUM_ALGSUB}(\*|/){RE_NUM_ALGSUB})", equation): # prod (x*y) and div (x/y)
        for whole,num0,alg0,operator,num1,alg1 in l:
            if alg0: x = _advanced_get_value(alg0, input_values, values, substitutions)
            else: x = float(num0)
            if alg1: z = _advanced_get_value(alg1, input_values, values, substitutions)
            else: z = float(num1)
            
            if operator=="/": x = x/z
            else: x = x*z
            
            if alg0 and alg1: # combined
                values[alg0] = x
                del values[alg1]
                equation = equation.replace(whole, alg0, 1)
            elif alg0:
                values[alg0] = x
                equation = equation.replace(whole, alg0, 1)
            elif alg1:
                values[alg1] = x
                equation = equation.replace(whole, alg1, 1)
            else: equation = equation.replace(whole, str(x), 1)
    return equation

def _advanced_operation_addsubtract(equation:str, input_values:dict, values:dict, substitutions:dict):
    while l:=re.findall(rf"({RE_NUM_ALGSUB}([\+\-]+){RE_NUM_ALGSUB})", equation): # add (x+y) and subtract (x-y)
        for whole,num0,alg0,operator,num1,alg1 in l:
            if alg0: x = _advanced_get_value(alg0, input_values, values, substitutions)
            else: x = float(num0)
            if alg1: z = _advanced_get_value(alg1, input_values, values, substitutions)
            else: z = float(num1)
            
            if operator.count("-")%2: x = x-z
            else: x = x+z
            
            if alg0 and alg1: # combined
                values[alg0] = x
                del values[alg1]
                equation = equation.replace(whole, alg0, 1)
            elif alg0:
                values[alg0] = x
                equation = equation.replace(whole, alg0, 1)
            elif alg1:
                values[alg1] = x
                equation = equation.replace(whole, alg1, 1)
            else: equation = equation.replace(whole, str(x), 1)
    return equation

def solve(equation, **input_values):
    equation = expand(equation)
##    print("solving", equation)
    eq, values, substitutions = _advanced_solve_setup(equation, **input_values)
    if eq is None: return
##    print("substitutions", eq)
    splits = list(split_equation(eq))
    
    def single(string, values, substitutions):
##        print("in", string)
        if m:=re.match(RE_FUNC_PARENTHESIS, string): prefix, inside = m.groups()
        else: prefix, inside = "", string
        
        # do operations
        for x,z in re.findall(RE_FACT, inside):
            inside = inside.replace(x, str(int(np.prod(np.arange(0, int(z))+1))))

        inside = _advanced_operation_negation(inside, input_values, values, substitutions)
        inside = _advanced_operation_power(inside, input_values, values, substitutions)
        inside = _advanced_operation_multdiv(inside, input_values, values, substitutions)
        inside = _advanced_operation_addsubtract(inside, input_values, values, substitutions)
        
        # no more operations to do
##        print("should have nothing to do", inside)
        num,alg = re.match(RE_NUM_ALGSUB, inside).groups()
        if alg: x = _advanced_get_value(alg, input_values, values, substitutions)
        else: x = float(num)
        if prefix:
            match prefix[-3:]:
                case "sin":
                    if prefix[0]=="a": x = np.asin(x)
                    else: x = np.sin(x)
                case "cos":
                    if prefix[0]=="a": x = np.acos(x)
                    else: x = np.cos(x)
                case "tan":
                    if prefix[0]=="a": x = np.atan(x)
                    else: x = np.tan(x)
                case "abs": x = abs(x)
                case "log": x = np.log10(x)
                case "ln": x = np.log(x)
                case _: print("unknown function prefix", prefix)
##            print(prefix, ":", x)
        if alg:
            values[alg] = x
            string = alg
        else: string = str(x)
##        print("out", string)
        return string
    
##    print("splits", splits)
    for index,split in enumerate(splits):
        eq = join_equation(split, splits)
        splits[index] = single(eq, values, substitutions)

    result = splits[-1]
    if result in values:
        return values[result]

    # otherwise there were no variables used in solving
    result = float(splits[-1])
    if int(result)==result: return int(result)
    return result










def derivative(equation, target="x"):
    equation = expand(equation)
    operations = list(order_of_operations(equation))
##    print(equation, "->", operations)
    results = []
    
    def has_target_inside(string):
        return bool(re.match(rf".*(?:{RE_OPS})?{target}(?:{RE_OPS})?.*", join_equation(string, operations)))

    def item_deriv(string):
        m = re.match(RE_NUM_ALGSUB, string)
        if m:
            num, alg = m.groups()
            if num or alg!=target: return "0"
            return "1" # alg == target
        return string

    def item_orig(string):
        return join_equation(string, operations)

    def process_operation(operation):
##        print("operation", operation)
        if m := re.match(rf"({RE_NUM_ALGSUB_NONCAP}|{RE_OPSUBS_NONCAP})({RE_OPS})({RE_NUM_ALGSUB_NONCAP}|{RE_OPSUBS_NONCAP})", operation):
            left, operator, right = m.groups()
            
            if operator[0]!="-" and "-" in operator:
                right = "-"*(operator.count("-")%2)+right
                operator = operator.rsplit("-", 1)[0]
                
            match operator:
                case "+": return f"{item_deriv(left)}+{item_deriv(right)}"
                case "-": return f"{item_deriv(left)}-{item_deriv(right)}"
                case "**":
                    y = has_target_inside(right)
                    if not y:
                        x = f"{right}*{item_deriv(left)}*{item_orig(left)}**({right}-1)"
                    else:
                        x = f"{item_orig(left)}**{item_orig(right)}"
                        x = f"{x}*({item_deriv(right)}*ln({item_orig(left)})+{item_deriv(left)}*({item_orig(right)})/{item_orig(left)})"
                    return x
                case "/":
                    x = has_target_inside(left)
                    y = has_target_inside(right)
                    if x or y:
                        return f"(({item_deriv(left)}*{item_orig(right)}-{item_deriv(right)}*{item_orig(left)})/({item_orig(right)}**2))"
                    return f"{left}/{right}"
                case "*":
                    return f"({item_deriv(right)}*{item_orig(left)}+{item_deriv(left)}*{item_orig(right)})"
        return item_deriv(operation)

    def process_inside(prefix, inside):
        result = process_operation(inside)
##        print("process_operation '", inside, "' result:", join_equation(result, results))
        suffix = ""
        if not prefix: result = f"({result})"
        else: # chain_rule in effect
            inside = join_equation(inside, operations)
            middle = inside
            suffix = ""
            match prefix:
                case "cos": prefix = "-sin"
                case "sin": prefix = "cos"
                case "tan":
                    prefix = "sec"
                    suffix = "**2"
                case "asin":
                    prefix = ""
                    middle = f"1/(1-({inside})**2)"
                case "acos":
                    prefix = ""
                    middle = f"-1/(1-({inside})**2)"
                case "atan":
                    prefix = ""
                    middle = f"1/(1+({inside})**2)"
                case "ln":
                    prefix = ""
                    middle = f"1/({inside})"
            result = f"{prefix}({middle}){suffix}*({result})"
        return result
        
    
    for operation in operations: # operations
        if not has_target_inside(operation): result = "0"
        elif m:=re.match(RE_FUNC_PARENTHESIS, operation):
            prefix, inside = m.groups()
            result = process_inside(prefix, inside)
        else:
            result = process_operation(operation)
        results.append(join_equation(result, results))
    
    if results[-1]: return results[-1]
    return "0"



class EquationObject():
    string = ""
    
    def __init__(self, string):
        self.string = string
    
    def required_inputs(self): # return a set of variables needed to solve
        return set(re.findall(RE_ALGEBRAIC, self.string)) - algebraic_blacklist

    def derivative(self, target_variable="x"):
        return type(self)(derivative(self.string, target=target_variable))

    def solve(self, **inputs):
        return solve(self.string, **inputs)
    
    def __str__(self): return self.string



class MappingFunction2D:
    def __init__(self):
        self.x = EquationObject("1-a*x**2+y")
        self.y = EquationObject("b*x*x")
        self.constants = {"a":1.4, "b":0.3}
        
    def required_constants(self): # return a set of keyword arguments __call__ requires
        need = self.x.required_inputs()
        need |= self.y.required_inputs()
        if "x" in need: need.remove("x")
        if "y" in need: need.remove("y")
        return need

    def missing_constants(self):
        return self.required_constants()-set(self.constants.keys())

    def trim_excess_constants(self):
        required = self.required_constants()
        for k in list(self.constants.keys()):
            if k not in required: del self.constants[k]

    def copy(self):
        new = type(self)()
        new.constants = self.constants.copy()
        new.x = self.x.copy()
        new.y = self.y.copy()
        return new
    
    def __str__(self):
        x = str(self.x)
        y = str(self.y)
        for k,v in self.constants.items():
            v = str(v)
            x = x.replace(k, v)
            y = y.replace(k, v)
        missing = self.missing_constants()
        return f"(x={x}, y={y})" + (str(" (has undefined constants)") if missing else "")

    def __call__(self, x, y, **inputs):
        return (self.x.solve(x=x, y=y, **inputs|self.constants),
                self.y.solve(x=x, y=y, **inputs|self.constants))

    def jacobian(self):
        return [
            [self.x.derivative("x"), self.x.derivative("y")],
            [self.y.derivative("x"), self.y.derivative("y")],
            ]

    def transposed_inverse_jacobian(self): # T(L)^-1
        jacobian = self.jacobian()
        # transpose
        jacobian = [[jacobian[0][0], jacobian[1][0]], [jacobian[0][1], jacobian[1][1]]] 

        # inverse & determinant
        det = f"({jacobian[0][0].string})*({jacobian[1][1].string})-({jacobian[0][1].string})*({jacobian[1][0].string})"
        jacobian = [[jacobian[1][1], jacobian[1][0]], [jacobian[0][1], jacobian[0][0]]]
        jacobian[0][1].string = f"-({jacobian[0][1].string})"
        jacobian[1][0].string = f"-({jacobian[1][0].string})"
        for i in range(2):
            for j in range(2):
                o = jacobian[i][j]
                o.string = f"({det})*({o.string})" # multiply with the determinant
        return jacobian

def test():
    input_values = {
        "x": np.linspace(0, 7, 5),
        "y": -8,
        "i": 5,
        "a": 1.4,
        "b": 0.1,
        }

    def single(eq):
        print(eq)
        eeq = expand(eq)
        print(eeq, "expand")
        seq = shrink(eq)
        print(seq, "shrink")
        eeq = expand(seq)
        print(eeq, "expand again")
        
        print(list(order_of_operations(eeq)), "expand order")
        print(list(order_of_operations(seq)), "shrink order")
        
        print(solve(eeq, **input_values), "expand solved")
        print(solve(seq, **input_values), "shrink solved")

        deeq = derivative(eeq, "x")
        print(deeq, "expand derivative")
        dseq = derivative(seq, "x")
        print(dseq, "shrink derivative")
        
        print(solve(deeq, **input_values), "expand derivative solved")
        print(solve(dseq, **input_values), "shrink derivative solved")
        print("")
        
    testables = [
##        "x**3",
##        "(x**3)**x",
##        "3**ln(x)",
##        "0**x+0*(1+x)",
##        "x**0+tan(1+x)*0",
##        "x**0+tan(1+x)**0+(x+1+(y**2)**0)**0",
##        "2/3*4**5+6**7.3*4*3+25+2(1)",
##        "1*(2+1-0.3)*x**3+8*0",
##        "2**(x-1)**1+2*1/3+0*x+y**0+2**0+(x+2(y-1))**0",
##        "(y*2+1)**(x-1+(60+4-(5)))",
##        "sin(2x**2+1)-cos(y/3)+tan(i)",
##        "x**2**3",
##        "sin(x**2)",
##        "cos(sin(x**2))",
##        "cos(x)/x+1/(x**2+3)^2",
##        "1+1",
##        "2*3",
##        "4**5.3",
##        "y**z",
##        "y**i",
##        "ln(x+8-5-10+9)",
##        "ln(8)",
##        "tan(8*6*-1*3)",
##        "x/(y)",
##        "1/(x+1)",
        
##        "x/2-y/3",
##        "y/3+x/3",
##        "sin(x)*y/3",
##        "y/3*sin(x)",
        "1-a*x**2+y",
        "b*x*x",
        
##        "(x**2)*(x+4)**-1",
##        "(x**2)/(x+4)",
##        "sin(x)*cos(x)+sin(x+1)+x*3+5*x",
##        "cos(x)*x",
##        "cos(sin(x+1))",
##        "cos(y*x+1)",
##        "6*x*x**2+2*x*y*x+x*x+x",
##        "-1+(6*x+1)*(6*x+1)**2+3",
        
##        "x**2/(x**3*y)",
##        "x**2/(x-(1/y))",
##        "-x**2",
##        "-x**-2.3",

##        "b*x*x", # OK
##        "1/x", # OK
##        "b*x*x^3+3/(y+x)", # OK
##        "-b*(x+1)*(x+1)^3-1", # OK
        ]
    for eq in testables: single(eq)





if __name__ == "__main__":
    test()

##    mf = MappingFunction2D()
####    jacob = mf.jacobian()
##    tij = mf.transposed_inverse_jacobian()
##    print(tij[0][0])
##    print(tij[0][1])
##    print(tij[1][0])
##    print(tij[1][1])

    
##    start_point = (0,0)
##    starting_points = np.repeat([start_point], radians.size, axis=0).astype(np.float64)




