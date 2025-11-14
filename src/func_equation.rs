//use regex::Regex;

pub fn find_parenthesis_start(string:&str, start_index:usize) -> usize {
    let mut closure:i64 = 0;
    for i in 0..start_index {
        let c = string.chars().nth(start_index-i);
        match c {
            Some('(') => closure += 1,
            Some(')') => closure -= 1,
            _ => {},
        }
        if closure>=0 {
            return start_index-i;
        }
    }
    return 0;
}

pub fn find_parenthesis_end(string:&str, start_index:usize) -> usize {
    let mut closure:i64 = 0;
    if string.chars().nth(start_index)==Some('(') {closure -= 1;}
    for i in 0..(string.len()-start_index) {
        let c = string.chars().nth(start_index+i);
        match c {
            Some('(') => closure += 1,
            Some(')') => closure -= 1,
            _ => {},
        }
        if closure<0 {
            return start_index+i;
        }
    }
    return string.len()-1;
}

const RE_INT:&str = r"(?:\d+)";//Regex::new().expect("???");
const RE_FLOAT:&str = r"(?:(?:\d*[\.\,]\d+)|(?:\d+[\.\,]\d*))";
const RE_NUMBER_POS:&str = r"(?:(?:\d*[\.\,]\d+)|(?:\d+(?:[\.\,]\d*)?))";
const RE_NUMBER:&str = r"\-?(?:(?:\d*[\.\,]\d+)|(?:\d+(?:[\.\,]\d*)?))";

const RE_ALGEBRAIC_POS:&str = r"(?:[a-zA-Z]+)";
const RE_ALGEBRAIC:&str = r"\-?(?:[a-zA-Z]+)";

const RE_SUBSTITUTION:&str = r"(?:\{\d+\})";
const RE_OPSUBS:&str = r"(?:\{\<(\d+)\>\})";
const RE_OPSUBS_NONCAP:&str = r"(?:\{\<\d+\>\})";

const RE_FACT:&str = r"(?:(\d+)\!+)";
const RE_FUNC:&str = r"(?:a?(?:sin|cos|tan)|abs|log|ln)";
const RE_ALGSUB:&str = r"(?:\-?(?:[a-zA-Z]+)|(?:\{\d+\}))";
const RE_ALGSUB_POS:&str = r"(?:(?:[a-zA-Z]+)|(?:\{\d+\}))";

const RE_FUNC_PARENTHESIS:&str = r"((?:a?(?:sin|cos|tan)|abs|log|ln))?\(([^\(\)]+)\)";
const RE_FUNC_PARENTHESIS_NONCAP:&str = r"((?:a?(?:sin|cos|tan)|abs|log|ln))?\([^\(\)]+\)";
const RE_NUM_ALGSUB:&str = r"(?:(\-?(?:(?:\d*[\.\,]\d+)|(?:\d+(?:[\.\,]\d*)?)))|((?:\-?(?:[a-zA-Z]+)|(?:\{\d+\}))))";
const RE_NUM_ALGSUB_NONCAP:&str = r"(?:(?:\-?(?:(?:\d*[\.\,]\d+)|(?:\d+(?:[\.\,]\d*)?)))|(?:(?:\-?(?:[a-zA-Z]+)|(?:\{\d+\}))))";
const RE_NUM_ALGSUB_NONCAP_POS:&str = r"(?:(?:(?:(?:\d*[\.\,]\d+)|(?:\d+(?:[\.\,]\d*)?)))|(?:(?:(?:[a-zA-Z]+)|(?:\{\d+\}))))";

const RE_OPERABLE:&str = r"(?:(\-?(?:(?:\d*[\.\,]\d+)|(?:\d+(?:[\.\,]\d*)?)))|((?:\-?(?:[a-zA-Z]+)|(?:\{\d+\}))|(?:\{\<\d+\>\})))";
const RE_OPERABLE_NONCAP:&str = r"(?:\-?(?:(?:\d*[\.\,]\d+)|(?:\d+(?:[\.\,]\d*)?))|(?:\-?(?:[a-zA-Z]+)|(?:\{\d+\}))|(?:\{\<\d+\>\}))";
const RE_OPERABLE_POS:&str = r"(?:((?:(?:\d*[\.\,]\d+)|(?:\d+(?:[\.\,]\d*)?)))|((?:(?:[a-zA-Z]+)|(?:\{\d+\}))|(?:\{\<\d+\>\})))";
const RE_OPERABLE_NONCAP_POS:&str = r"(?:(?:(?:\d*[\.\,]\d+)|(?:\d+(?:[\.\,]\d*)?))|(?:(?:[a-zA-Z]+)|(?:\{\d+\}))|(?:\{\<\d+\>\}))";

const POWER_PATTERN:&str = r"(((?:(?:(?:\D*[\.\,]\D+)|(?:\D+(?:[\.\,]\D*)?))|(?:(?:[A-ZA-Z]+)|(?:\{\D+\}))|(?:\{\<\D+\>\})))(?:\*\*|\^)\-?((?:(?:(?:\D*[\.\,]\D+)|(?:\D+(?:[\.\,]\D*)?))|(?:(?:[A-ZA-Z]+)|(?:\{\D+\}))|(?:\{\<\D+\>\}))))";
const DIVISION_PATTERN:&str = r"(((?:(?:(?:\D*[\.\,]\D+)|(?:\D+(?:[\.\,]\D*)?))|(?:(?:[A-ZA-Z]+)|(?:\{\D+\}))|(?:\{\<\D+\>\})))/\-?((?:(?:(?:\D*[\.\,]\D+)|(?:\D+(?:[\.\,]\D*)?))|(?:(?:[A-ZA-Z]+)|(?:\{\D+\}))|(?:\{\<\D+\>\}))))";
const MULTIPLY_PATTERN:&str = r"(((?:(?:(?:\D*[\.\,]\D+)|(?:\D+(?:[\.\,]\D*)?))|(?:(?:[A-ZA-Z]+)|(?:\{\D+\}))|(?:\{\<\D+\>\})))\*\-?((?:(?:(?:\D*[\.\,]\D+)|(?:\D+(?:[\.\,]\D*)?))|(?:(?:[A-ZA-Z]+)|(?:\{\D+\}))|(?:\{\<\D+\>\}))))";
const ADDITION_PATTERN:&str = r"(((?:(?:(?:\D*[\.\,]\D+)|(?:\D+(?:[\.\,]\D*)?))|(?:(?:[A-ZA-Z]+)|(?:\{\D+\}))|(?:\{\<\D+\>\})))\++((?:(?:(?:\D*[\.\,]\D+)|(?:\D+(?:[\.\,]\D*)?))|(?:(?:[A-ZA-Z]+)|(?:\{\D+\}))|(?:\{\<\D+\>\}))))";
const SUBTRACT_PATTERN:&str = r"(((?:(?:(?:\D*[\.\,]\D+)|(?:\D+(?:[\.\,]\D*)?))|(?:(?:[A-ZA-Z]+)|(?:\{\D+\}))|(?:\{\<\D+\>\})))\+*\-\+*((?:(?:(?:\D*[\.\,]\D+)|(?:\D+(?:[\.\,]\D*)?))|(?:(?:[A-ZA-Z]+)|(?:\{\D+\}))|(?:\{\<\D+\>\}))))";
const NEGATION_PATTERN:&str = r"((?:\-)((?:(?:(?:\D*[\.\,]\D+)|(?:\D+(?:[\.\,]\D*)?))|(?:(?:[A-ZA-Z]+)|(?:\{\D+\}))|(?:\{\<\D+\>\}))))";

const RE_OPS:&str = r"[\*/\+\-\^]+";

const ALGEBRAIC_BLACKLIST:&'static [&'static str] = &["sin","cos","tan","asin","acos","atan", "abs", "log", "ln"];



pub fn create_substitutions(equation:&str) -> &str {
    return equation;
}



pub fn testing() {
    println!("{}", find_parenthesis_start("hel(lo)qwe", 6));
    println!("{}", find_parenthesis_end("hel(lo)qwe", 3));
    println!("{}", NEGATION_PATTERN);
    println!("{}", ALGEBRAIC_BLACKLIST[0]);
}

