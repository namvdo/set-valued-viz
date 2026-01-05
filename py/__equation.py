import re
import numpy as np


def find_parenthesis_start(string, start_index, start_symbol="(", end_symbol=")"):
    index = 0
    closure = 0
    for i in range(start_index):
        c = string[start_index-i]
        if c==start_symbol: closure += 1
        elif c==end_symbol: closure -= 1
        if closure>=0:
            index = start_index-i
            break
    return index
def find_parenthesis_end(string, start_index, start_symbol="(", end_symbol=")"):
    index = len(string)-1
    closure = -int(string[start_index]==start_symbol)
    for i in range(len(string)-start_index):
        c = string[start_index+i]
        if c==start_symbol: closure += 1
        elif c==end_symbol: closure -= 1
        if closure<0:
            index = start_index+i
            break
    return index


RE_INT = r"(?:\d+)"
RE_FLOAT = r"(?:(?:\d*[\.\,]\d+)|(?:\d+[\.\,]\d*))"

RE_NUMBER_POS = r"(?:(?:\d*[\.\,]\d+)|(?:\d+(?:[\.\,]\d*)?))"#rf"(?:{RE_FLOAT}|{RE_INT})"
RE_NUMBER = rf"\-?{RE_NUMBER_POS}"

RE_ALGEBRAIC_POS = r"(?:[a-zA-Z]+)"
RE_ALGEBRAIC = rf"\-?{RE_ALGEBRAIC_POS}"

RE_SUBSTITUTION = r"(?:\{\d+\})"
RE_OPSUBS = r"(?:\{\<(\d+)\>\})"
RE_OPSUBS_NONCAP = r"(?:\{\<\d+\>\})"

RE_FACT = r"(?:(\d+)\!+)"
RE_FUNC = r"(?:a?(?:sin|cos|tan)|abs|log|ln)"
RE_ALGSUB = rf"(?:{RE_ALGEBRAIC}|{RE_SUBSTITUTION})"
RE_ALGSUB_POS = rf"(?:{RE_ALGEBRAIC_POS}|{RE_SUBSTITUTION})"

RE_FUNC_PARENTHESIS = rf"({RE_FUNC})?\(([^\(\)]+)\)"
RE_FUNC_PARENTHESIS_NONCAP = rf"(?:{RE_FUNC})?\([^\(\)]+\)"
RE_NUM_ALGSUB = rf"(?:({RE_NUMBER})|({RE_ALGSUB}))"
RE_NUM_ALGSUB_NONCAP = rf"(?:(?:{RE_NUMBER})|(?:{RE_ALGSUB}))"
RE_NUM_ALGSUB_NONCAP_POS = rf"(?:(?:{RE_NUMBER_POS})|(?:{RE_ALGSUB_POS}))"

RE_OPERABLE = rf"(?:({RE_NUMBER})|({RE_ALGSUB}|{RE_OPSUBS_NONCAP}))"
RE_OPERABLE_NONCAP = rf"(?:{RE_NUMBER}|{RE_ALGSUB}|{RE_OPSUBS_NONCAP})"
RE_OPERABLE_POS = rf"(?:({RE_NUMBER_POS})|({RE_ALGSUB_POS}|{RE_OPSUBS_NONCAP}))"
RE_OPERABLE_NONCAP_POS = rf"(?:{RE_NUMBER_POS}|{RE_ALGSUB_POS}|{RE_OPSUBS_NONCAP})"

POWER_PATTERN = rf"(({RE_OPERABLE_NONCAP_POS})(?:\*\*|\^)\-?({RE_OPERABLE_NONCAP_POS}))"
DIVISION_PATTERN = rf"(({RE_OPERABLE_NONCAP_POS})/\-?({RE_OPERABLE_NONCAP_POS}))"
MULTIPLY_PATTERN = rf"(({RE_OPERABLE_NONCAP_POS})\*\-?({RE_OPERABLE_NONCAP_POS}))"
ADDITION_PATTERN = rf"(({RE_OPERABLE_NONCAP_POS})\++({RE_OPERABLE_NONCAP_POS}))"
SUBTRACT_PATTERN = rf"(({RE_OPERABLE_NONCAP_POS})\+*\-\+*({RE_OPERABLE_NONCAP_POS}))"
NEGATION_PATTERN = rf"((?:\-)({RE_OPERABLE_NONCAP_POS}))"

RE_OPS = r"[\*/\+\-\^]+"
RE_USELESS_PARENTHESIS = rf"(({RE_FUNC}|^)?\(({RE_NUM_ALGSUB_NONCAP})\))" # detect useless parenthesis

ALGEBRAIC_BLACKLIST = {"sin","cos","tan","asin","acos","atan", "abs", "log", "ln"} # invalid variable names in equations

RE_EQJOINABLE = rf"(?:.*|^)({RE_OPSUBS})(?:.*|$)"

def create_substitutions(equation):
    i = 0
    substitutions = {}
    string_index = 0
    for k in re.findall(RE_NUMBER_POS, equation): # replace numbers with operable objects
        a = "{"+str(i)+"}"
        i += 1
        substitutions[a] = k
        string_index1 = equation[string_index:].index(k)
        equation = equation[:string_index]+equation[string_index:].replace(k, a, 1)
        string_index += string_index1+len(a)
    
    string_index = 0
    for k in re.findall(RE_ALGEBRAIC_POS, equation):
        if k not in ALGEBRAIC_BLACKLIST:
            a = "{"+str(i)+"}"
            i += 1
            substitutions[a] = k
            string_index1 = equation[string_index:].index(k)
            equation = equation[:string_index]+equation[string_index:].replace(k, a, 1)
            string_index += string_index1+len(a)
        else: string_index += len(k)
    return equation, substitutions

def fill_in_substitutions(equation, substitutions):
    for whole in re.findall(RE_SUBSTITUTION, equation):
        if whole in substitutions:
            equation = equation.replace(whole, substitutions[whole])
    return equation



def order_of_operations(equation):
    pattern_order_check_negation = [POWER_PATTERN,DIVISION_PATTERN,MULTIPLY_PATTERN]
    pattern_order_do_not_check = [SUBTRACT_PATTERN,NEGATION_PATTERN,ADDITION_PATTERN]
    
    index = 0
    equation = "("+equation+")"
    while closures:=re.findall(rf"((?:{RE_FUNC})?\(([^\(\)]+)\))", equation):
        for closure,inside in closures:
            old_inside = inside
            for pattern in pattern_order_check_negation:
                while operations:=re.findall(pattern, inside):
                    for operation in operations:
                        operation = operation[0]
                        old_operation = operation
                        while negations:=re.findall(NEGATION_PATTERN, operation):
                            for negation in negations:
                                negation = negation[0]
                                yield negation
                                operation = operation.replace(negation, "{<"+str(index)+">}", 1)
                                index += 1
                        yield operation
                        inside = inside.replace(old_operation, "{<"+str(index)+">}", 1)
                        index += 1
            
            for pattern in pattern_order_do_not_check:
                while operations:=re.findall(pattern, inside):
                    for operation in operations:
                        operation = operation[0]
                        yield operation
                        inside = inside.replace(operation, "{<"+str(index)+">}", 1)
                        index += 1
            
            yield closure.replace(old_inside, inside)
            equation = equation.replace(closure, "{<"+str(index)+">}", 1)
            index += 1
    yield equation

def join_equation(equation, parts):
    while m:=re.match(RE_EQJOINABLE, equation):
        equation = equation.replace(m.group(1), parts[int(m.group(2))])
    return equation


def remove_useless_ones(eq): # "**1" & "/1"
    for whole,target,end in re.findall(r"(((?:\*\*|\*|/|\^)1)([\*/\)]|$|\D))", eq):
##        print("removing one", whole, "in", eq)
        i = eq.index(whole)
        eq = eq[:i]+eq[i:].replace(target, "", 1)
    for whole,start,target in re.findall(r"((\(|^|\*)(1\*)[^\*])", eq):
##        print("removing one", whole, "in", eq)
        i = eq.index(whole)+len(start)
        eq = eq[:i]+eq[i:].replace(target, "", 1)
    return eq

def remove_useless_zeros(eq):
    for whole,target,end in re.findall(r"((?:\(|^)(\-+0+)([\+\-\)]|$))", eq):
        i = eq.index(whole)
        eq = eq[:i]+eq[i:].replace(target, "0", 1)
    while found:=re.findall(r"(([\+\-]+0+)([\+\-\)]|$))", eq):
        for whole,target,end in found:
            i = eq.index(whole)
            eq = eq[:i]+eq[i:].replace(target, "", 1)
    while found:=re.findall(r"(([\+\-\(]|^)(0+[\+\-]+))", eq):
        for whole,start,target in found:
            i = eq.index(whole)+len(start)
            eq = eq[:i]+eq[i:].replace(target, "-"*(target[-1]=="-" and start!="-"), 1)
    return eq

def shrink_power_symbols(eq):
    pattern = rf"(({RE_NUM_ALGSUB_NONCAP}|\))\*\*({RE_NUM_ALGSUB_NONCAP}|\())"
    for whole,x,y in re.findall(pattern, eq): eq = eq.replace(whole, f"{x}^{y}")
    return eq

def remove_useless_product_symbols(eq):
    # remove useless product symbols
    pattern = rf"(({RE_NUMBER}|[\(\)])\*({RE_ALGSUB_POS}|[\(\)]))"
    for whole,x,y in re.findall(pattern, eq): eq = eq.replace(whole, f"{x}{y}")
    return eq

def remove_useless_parenthesis(eq):
    while m:=re.match(r".*(\(\(([^\)\(]*)\)\)).*", eq):
        eq = eq.replace(m.group(1), f"({m.group(2)})")
    
    for whole,f,inside in re.findall(RE_USELESS_PARENTHESIS, eq):
        if not f:
            i = eq.index(whole)
            eq = eq[:i]+eq[i:].replace(whole, inside, 1)
    return eq

def fix_too_many_additions(eq):
    for whole,start,minuses in re.findall(r"((\(|^)?([\-\+]{2,}))", eq):
        replacement = "-" if len(minuses)%2 else ("+" if start is None else "")
        i = eq.index(whole)
        eq = eq[:i]+eq[i:].replace(whole, start+replacement, 1)
    return eq

def shrink(eq):
    orig_eq = eq
    eq = fix_too_many_additions(eq)
    eq = re.sub(r"\s", "", eq) # remove whitespace
    
    symbol_p = r"(?:\*\*|\^)"
    pattern1 = rf"((?:\(|\+|^|{RE_ALGSUB})(\-?(?:{RE_NUMBER_POS}[\+\-]+)+(?:{RE_NUMBER_POS}))(?:\)|$|{RE_ALGSUB}))" # trivial addition operations
    pattern3 = rf"(((?:{RE_NUM_ALGSUB_NONCAP_POS}|\)){symbol_p}\-?0)(?:{RE_OPS}|$|\)))" # zero power
    pattern2 = rf"((?:([\+\-]|\(|^)0+(?:\*+|/)(?:{RE_NUM_ALGSUB_NONCAP}|\())|(?:(?:{RE_NUM_ALGSUB_NONCAP}|\))\*\-?0+([\+\-]|\)|$)))" # zero multipliers
    pattern4 = rf"((?:\(|^|([\^\*][\*\+\-]))((?:{RE_NUMBER}\*)+(?:{RE_NUMBER}))(?:\)|$|[/\+\-]|{RE_ALGSUB}))" # trivial product operations
    pattern5 = rf"((?:\(|^|([\*\+\-/]))((?:{RE_NUMBER}{symbol_p})+(?:{RE_NUMBER}))(?:\)|$|[/\+\-]|[^{symbol_p}]|{RE_ALGSUB}))" # trivial power operations
    
    done = False
    while not done:
        eq = remove_useless_parenthesis(eq)
##        print(eq)
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
                    negative = 0
            if int(total)==total: total = int(total)
            
            i = eq.index(whole)
            eq = eq[:i]+eq[i:].replace(inside, "+"*int(total>=0 and inside[0]=="-")+str(total), 1)

        results3 = re.findall(pattern3, eq)
        for whole,target in results3:
##            print("oneing", whole)
            if target[0]==")":
                end_i = eq.index(whole)
                start_i = find_parenthesis_start(eq, end_i)
                end_i += len(target)
                start_i -= len(re.match(r"(?:.*[^a-zA-Z]|^)([a-zA-Z]*)$", eq[:start_i])[1]) # remove prefix len aswell
                eq = eq[:start_i]+"1"+eq[end_i:]
            else:
                i = eq.index(whole)
                eq = eq[:i]+"1"+eq[i+len(target):]
        
        results2 = re.findall(pattern2, eq)
        for whole,prefix,suffix in results2:
            if whole not in eq: continue
            zero = prefix+"0"+suffix
##            print("zeroing", whole, "in", eq, "replacement:", zero)
            if whole[0]==")":
                end_i = eq.index(whole)
                start_i = find_parenthesis_start(eq, end_i)
                end_i += len(whole)
                start_i -= len(re.match(r"(?:.*[^a-zA-Z]|^)([a-zA-Z]*)$", eq[:start_i])[1]) # remove prefix len aswell
                eq = eq[:start_i]+zero+eq[end_i:]
            elif whole[-1]=="(":
                start_i = eq.index(whole)
                end_i = find_parenthesis_end(eq, start_i+len(whole)-1)
                eq = eq[:start_i]+zero+eq[end_i+1:]
            else:
                i = eq.index(whole)
                eq = eq[:i]+eq[i:].replace(whole, zero, 1)

        results5 = re.findall(pattern5, eq)
        for whole,separating_op,inside in results5:
            total = None
            for x in re.split(symbol_p, inside):
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
        
        if not results1 and not results2 and not results3 and not results4 and not results5:
            done = True
    
    eq = remove_useless_parenthesis(eq)
    eq = remove_useless_ones(eq)
    eq = remove_useless_zeros(eq)
    eq = shrink_power_symbols(eq)
    eq = remove_useless_product_symbols(eq)
    
    if not eq or eq=="()": return "0"
    return eq



def expand(eq):
    not_op0 = r"[^\-\+\^\*/\(]"
    not_op1 = r"[^\-\+\^\*/\)]"
    
    # restore default power symbols
    pattern = rf"(({not_op0})\^({not_op1}))"
    for whole,x,y in re.findall(pattern, eq):
        eq = eq.replace(whole, f"{x}**{y}")
    
    # restore product symbol
    pattern = rf"((?:({RE_FUNC})|({RE_NUMBER}|\)))({RE_FUNC}|{RE_ALGSUB_POS}|\())"
    for whole,f,x,y in re.findall(pattern, eq):
        if f: continue
        i = eq.index(whole)+len(x)
        eq = eq[:i]+"*"+eq[i:]
    return eq


def power(x, z):
    return x**z
def multiply(x, z):
    return x*z
def division(x, z):
    if type(z)==np.ndarray:
        result = np.zeros_like(z)
        valid = z!=0
        if type(x)==np.ndarray: result[valid] = x[valid]/z[valid]
        else: result[valid] = x/z[valid]
        result[~valid] = np.nan
        return result
    if z==0: return np.nan
    return x/z
def subtract(x, z):
    return x-z
def addition(x, z):
    return x+z
def negation(x):
    return -x

def solve(equation, **input_values):
    equation = expand(shrink(equation))
    
    values = {}
    equation, substitutions = create_substitutions(equation)
    
    def get_value(key):
        if key not in values and key in substitutions:
            k = substitutions[key]
            if k in input_values:
                if hasattr(input_values[k], "copy"): values[key] = input_values[k].copy()
                else: values[key] = input_values[k]
            else: values[key] = float(k)
            return values[key]
        return values.get(key, key)
        
    def two_term_operations(pattern, string, func):
        while m:=re.match(pattern, string):
            whole,sub0,sub1 = m.groups()
            values[sub0] = func(get_value(sub0), get_value(sub1))
            del values[sub1]
            string = string.replace(whole, sub0, 1)
        return string

    def one_term_operations(pattern, string, func):
        while m:=re.match(pattern, string):
            whole,sub0 = m.groups()
            values[sub0] = func(get_value(sub0))
            string = string.replace(whole, sub0, 1)
        return string
    
    def single(string, values, substitutions):
        if m:=re.match(RE_FUNC_PARENTHESIS, string): prefix, inside = m.groups()
        else: prefix, inside = "", string
        
        # do operations
        for x,z in re.findall(RE_FACT, inside): # factorials
            inside = inside.replace(x, str(int(np.prod(np.arange(0, int(z))+1))))
        
        if m:=re.match(r"^(\-{2,}).*", inside): inside = inside.replace(m.group(1), "-"*(len(m.group(1))%2))
        
        if out := two_term_operations(POWER_PATTERN, inside, power):
            inside = out
        if out := two_term_operations(DIVISION_PATTERN, inside, division):
            inside = out
        if out := two_term_operations(MULTIPLY_PATTERN, inside, multiply):
            inside = out
        if out := two_term_operations(SUBTRACT_PATTERN, inside, subtract):
            inside = out
        if out := one_term_operations(NEGATION_PATTERN, inside, negation):
            inside = out
        if out := two_term_operations(ADDITION_PATTERN, inside, addition):
            inside = out
        
        # no more operations to do
        x = get_value(inside)
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
                case "log":
                    if type(x)==np.ndarray:
                        valid = (x>0)*(x<np.inf)
                        x[valid] = np.log10(x[valid])
                        x[~valid] = np.nan
                    else:
                        if x>0 and x<np.inf: x = np.log10(x)
                        else: x = np.nan
                case "ln":
                    if type(x)==np.ndarray:
                        valid = (x>0)*(x<np.inf)
                        x[valid] = np.log(x[valid])
                        x[~valid] = np.nan
                    else:
                        if x>0 and x<np.inf: x = np.log(x)
                        else: x = np.nan
                case _: print("unknown function prefix", prefix)
        values[inside] = x
        return inside
    
    results = []
    for index,op in enumerate(order_of_operations(equation)):
        op = join_equation(op, results)
        op_result = single(op, values, substitutions)
        results.append(op_result)
    result = results[-1]
    
    if result in values: return values[result]
    result = float(result) # otherwise there were no variables used in solving
    if int(result)==result: return int(result)
    return result




def derivative(equation, target="x"):
    equation = expand(shrink(equation))
    operations = list(order_of_operations(equation))
    results = []
    
    def has_target_inside(string):
        return bool(re.match(rf".*(?:{RE_OPS})?{target}(?:{RE_OPS})?.*", join_equation(string, operations)))

    def item_deriv(string):
        if m:=re.match(RE_NUM_ALGSUB, string):
            num, alg = m.groups()
            if alg and has_target_inside(alg): return "1"
            return "0"
        return string

    def item_orig(string):
        return join_equation(string, operations)

    def process_operation(operation):
        if m := re.match(rf"({RE_OPERABLE_NONCAP_POS})({RE_OPS})({RE_OPERABLE_NONCAP_POS})", operation):
            left, operator, right = m.groups()
            
            if operator=="^" or operator=="**":
                y = has_target_inside(right)
                if not y:
                    x = f"{item_orig(right)}*{item_deriv(left)}*{item_orig(left)}**({item_orig(right)}-1)"
                else:
                    x = f"{item_orig(left)}**{item_orig(right)}"
                    x = f"{x}*({item_deriv(right)}*ln({item_orig(left)})+{item_deriv(left)}*({item_orig(right)})/{item_orig(left)})"
                return x
            
            match operator:
                case "+": return f"{item_deriv(left)}+{item_deriv(right)}"
                case "-": return f"{item_deriv(left)}-{item_deriv(right)}"
                case "/":
                    x = has_target_inside(left)
                    y = has_target_inside(right)
                    if x or y:
                        return f"(({item_deriv(left)}*{item_orig(right)}-{item_deriv(right)}*{item_orig(left)})/({item_orig(right)}**2))"
                    return f"{item_orig(left)}/{item_orig(right)}"
                case "*":
                    return f"({item_deriv(right)}*{item_orig(left)}+{item_deriv(left)}*{item_orig(right)})"
        return item_deriv(operation)

    def process_inside(prefix, inside):
        result = "("+process_operation(inside)+")"
        if prefix: # chain rule in effect
            inside_derivative = result
            inside = join_equation(inside, operations) # freeze as original form
            match prefix:
                case "cos": result = f"-sin({inside})"
                case "sin": result = f"cos({inside})"
                case "tan": result = f"sec({inside})**2"
                case "asin": result = f"1/(1-({inside})**2)"
                case "acos": result = f"-1/(1-({inside})**2)"
                case "atan": result = f"1/(1+({inside})**2)"
                case "ln": result = f"1/({inside})"
            result = f"({inside_derivative}*{result})"
        return result
        
    
    for operation in operations: # operations
        if not has_target_inside(operation): result = "0"
        elif m:=re.match(RE_FUNC_PARENTHESIS, operation):
            prefix, inside = m.groups()
            result = process_inside(prefix, inside)
##            print("process_inside '", (prefix, inside), "' result:", join_equation(result, results))
        else:
            result = process_operation(operation)
        result = join_equation(result, results)
        result = expand(shrink(result))
##        print("process ->", operation, "result:", result)
        results.append(result)
    
    if results[-1]: return expand(shrink(results[-1]))
    return "0"







class EquationObject():
    def __init__(self, string):
        self.string = string
    
    def required_inputs(self): # return a set of variables needed to solve
        return set(re.findall(RE_ALGEBRAIC_POS, self.string)) - ALGEBRAIC_BLACKLIST

    def derivative(self, target_variable="x"):
        return type(self)(derivative(self.string, target=target_variable))

    def solve(self, **inputs):
        return solve(self.string, **inputs)
    
    def copy(self): return type(self)(self.string)
    
    def __str__(self): return self.string




def transpose_matrix(matrix):
    l = len(matrix)
    for i in range(0, l):
        for j in range(i+1, l):
            temp = matrix[i][j]
            matrix[i][j] = matrix[j][i]
            matrix[j][i] = temp


def solve_matrix(matrix, **inputs):
    output = []
    l = len(matrix)
    for i in range(l):
        temp = []
        for j in range(l):
            result = np.asarray(matrix[i][j].solve(**inputs))
            np.nan_to_num(result, copy=False)
            temp.append(result)
        output.append(temp)
    return output


class MappingFunction: # N-dimensional version of MappingFunction2D
    def __init__(self, **functions):
        self.constants = {}
        self.symbols = []
        for symbol,function_string in functions.items():
            self.add(symbol, function_string)
    
    def add(self, symbol, function_string):
        if not hasattr(self, symbol):
            f = EquationObject(function_string)
            setattr(self, symbol, f)
            self.symbols.append(symbol)
            self.constants[symbol] = 0 # placeholder value

    def remove(self, symbol):
        if symbol in self.symbols:
            delattr(self, symbol)
            self.symbols.remove(symbol)

    def get(self, index):
        k = self.symbols[index]
        return k, getattr(self, k)
    
    def __len__(self):
        return len(self.symbols)

    def __iter__(self):
        for k in self.symbols:
            yield k, getattr(self, k)
        
    def set_constants(self, **values):
        self.constants.update(values)
        
    def required_constants(self): # return a set of keyword arguments __call__ requires
        need = set()
        for k,f in self: need |= f.required_inputs()
        for k in self.symbols:
            if k in need: need.remove(k)
        return need

    def missing_constants(self):
        return self.required_constants()-set(self.constants.keys())

    def trim_excess_constants(self):
        required = self.required_constants()
        for k in list(self.constants.keys()):
            if k not in required: del self.constants[k]

    def copy(self):
        new = type(self)(**{k:f.string for k,f in self})
        new.constants = self.constants.copy()
        return new

    def __str__(self):
        strings_dict = {k:f.string for k,f in self}
        
        for k,v in self.constants.items():
            v = str(v)
            for kk,vv in strings_dict.items():
                strings_dict[kk] = vv.replace(k, v)
        string = "MappingFunction("
        i = 0
        for k,v in strings_dict.items():
            if i!=0: string += ", "
            string += f"{k} = {v}"
            i += 1
        if bool(self.missing_constants()):
            string += ", undefined constants"
        string += ")"
        return string

    def __call__(self, **inputs):
        outputs = []
        for k,f in self:
            outputs.append(f.solve(**self.constants|inputs))
        return outputs
    
    def jacobian(self): # full matrix
        matrix = []
        l = len(self)
        for i in range(l):
            k, f = self.get(i)
            row = []
            for j in range(l):
                row.append(f.derivative(self.symbols[j]))
            matrix.append(row)
        return matrix
    
    def jacobian2D(self): # limit to first 2 dimensions
        matrix = []
        l = min(len(self), 2)
        for i in range(l):
            k, f = self.get(i)
            row = []
            for j in range(l):
                row.append(f.derivative(self.symbols[j]))
            matrix.append(row)
        return matrix
    
    def jacobian3D(self): # limit to first 3 dimensions
        matrix = []
        l = min(len(self), 3)
        for i in range(l):
            k, f = self.get(i)
            row = []
            for j in range(l):
                row.append(f.derivative(self.symbols[j]))
            matrix.append(row)
        return matrix

    def tij2D(self): # transposed_inverse_jacobian
        # T(L)^-1
        l = len(self)
        if l<2: return None
        matrix = self.jacobian2D()
        transpose_matrix(matrix)
        
        # determinant
        det = f"({matrix[0][0].string})*({matrix[1][1].string})-({matrix[0][1].string})*({matrix[1][0].string})"
        det = expand(shrink(det))
        if det=="0":
            print("non invertible matrix")
            return None
        
        # inverse
        matrix[0][0], matrix[1][1] = matrix[1][1], matrix[0][0]
        matrix[0][1].string = f"-({matrix[0][1].string})"
        matrix[1][0].string = f"-({matrix[1][0].string})"
        for i in range(2):
            for j in range(2):
                o = matrix[i][j]
                o.string = f"({o.string})/({det})" # divide with the determinant
        return matrix
        
    
    def tij3D(self): # transposed_inverse_jacobian
        # T(L)^-1
        l = len(self)
        if l<3: return None
        matrix = self.jacobian3D()
        transpose_matrix(matrix)
        
        # determinant
        #   using "rule of sarrus" -> only works with 3x3 matrices
        det = ""
        for i in range(l*2):
            temp = ""
            for j in range(l):
                if len(temp)!=0: temp += "*"
                if i>=l: j = -j
                temp += "("+matrix[(i+j)%l][j].string+")"
                
            if i>=l: det += "-"
            elif i>0: det += "+"
            det += temp
        det = expand(shrink(det))
        if det=="0":
            print("non invertible matrix")
            return None
        
        # inverse using the determinant
        transpose_matrix(matrix)
        for i in range(l):
            for j in range(l):
                o = matrix[i][j]
                o.string = f"({o.string})/({det})" # divide with the determinant
        return matrix





if __name__ == "__main__":
    points = np.random.random((3,2))

    fx = "x*2-a"
    fy = "y**2"
    
    mf = MappingFunction(x=fx, y=fy, z="x*y")
    print(mf)
    
    print(points)
    outputs = mf(x=points[:,0], y=points[:,1], a=.1) # , z=points[:,2]
    print(outputs)
    points[:,0] = outputs[0]
    points[:,1] = outputs[1]
    
    for row in mf.tij2D(): print(row)
    for k,f in mf: print(k, f)
    
##    input_values = {
##        "x": np.linspace(-1,1,11),
##        "y": np.linspace(-1,1,11),
##        "i": 5,
##        "a": 1.4,
##        "b": 0.1,
##        }
##    
##    def single(orig, deriv, target="x"):
##        print("derivative of", orig)
##        print("should be equal to", deriv)
##        orig_deriv = derivative(orig, target)
##        print("is actually", orig_deriv)
##        solve_test(orig_deriv)
##        solve_test(deriv)
##        print("\n"*2)
##
##    def solve_test(eq1):
##        print("\nsolving", eq1)
##        r1 = solve(eq1, **input_values)
##        r1 = np.nan_to_num(r1)
##        print("solved", r1)
##    
##    testables = [
####        ("x**2", "2*x"),
####        ("x**3", "3*x**2"),
####        ("(x**3)**x", "(x**3)**x*(ln(x**3)+3)"),
####        ("3**ln(x)", "(ln(3)*3**ln(x))/x"),
####        ("0**x+0*(1+x)", "0"),
####        ("2**(x-1)**1+2*1/3+0*x+y**0+2**0+(x+2(y-1))**0", "2**(x-1)*ln(2)"),
####        ("(y*2+1)**(x-1+(60+4-(5)))", "(2*y+1)**(x+58)*ln(2*y+1)"),
####        ("sin(2x**2+1)-cos(y/3)+tan(i)", "4*x*cos(2*x**2+1)"),
####        ("x**2**3", "8*x**7"),
####        ("sin(x**2)", "2*x*cos(x**2)"),
####        ("b*x*x", "2*b*x"),
##        ("1-a*x*x+y", "-2*a*x"),
##        ("-(-(1-a*x**2+y))", "-2*a*x"),
##        
####        ("sin(x)*cos(x)+sin(x+1)+x*3+5*x", "-(sin(x)^2) + cos(x)^2 + cos(x + 1) + 8"),
####        ("x/10", "1/10"),
####        ("x*x*y*x", "3*x*x*y"),
####        ("x/2+cos(x)**2", "-2*sin(x)*cos(x)+1/2"),
####        ("y/2-cos(x)*2", "sin(x)*2"),
####        ("cos(x)*-sin(x)*-cos(x)", "cos(x)**3-2*sin(x)**2*cos(x)"),
####        ("12*x*-0**0", "12"),
####        ("-(-(x+1))", "1"),
####        ("-x**-2.3", "-2.3*-x**-3.3"),
##        ]
##    for eq,deq in testables:
##        solve_test(eq)
##        single(eq,deq)
##    
##    print(input_values)
