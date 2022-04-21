use std::collections::{HashMap, HashSet};

extern crate regex;
use regex::Regex;

/// Handle message with tokenizing
pub fn tokenize(lower_case_text: &str) -> HashSet<&str> {
    Regex::new(r"[a-z0-9']+")
        .unwrap()
        .find_iter(lower_case_text)// non-overlapping
        .map(|mat| mat.as_str())
        .collect()
}

#[derive(Debug, Copy, Clone, Default)]
pub struct Message<'a> {
    pub text: &'a str,
    pub is_spam: bool,
}

impl<'a> Message<'a> {
    pub fn new(text: &'a str) -> Self {
        Self {
            text,
            is_spam: false,
        }
    }

    pub fn is_spam(&self) -> bool {
        self.is_spam
    }
}

/// Laplace Smoothing
// Design pattern: Use object but reference 
#[derive(Debug, Clone, Default)]
pub struct NativeBayesClassifier{
    pub alpha: f64,
    pub tokens: HashSet<String>,
    pub token_ham_counts: HashMap<String, i32>,
    pub token_spam_counts: HashMap<String, i32>,
    pub spam_messages_count: i32,
    pub ham_messages_count: i32,
}

impl NativeBayesClassifier {
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            tokens: HashSet::new(),
            token_ham_counts: HashMap::new(),
            token_spam_counts: HashMap::new(),
            spam_messages_count: 0,
            ham_messages_count: 0,
        }
    }

    pub fn train(&mut self, messages: &[Message]) {
        for  msg in messages {
            self.incre_msg_cl_count(msg);
            for token in tokenize(&msg.text.to_lowercase()) {
                self.tokens.insert(token.to_string());
                self.incre_token_count(token, msg.is_spam());
            }
        }
    }
    
    fn incre_token_count(&mut self, token: &str, is_spam: bool) {
        if !self.token_spam_counts.contains_key(token) {
            self.token_spam_counts.insert(token.to_string(), 0);
        }

        if !self.token_ham_counts.contains_key(token) {
            self.token_ham_counts.insert(token.to_string(), 0);
        }

        if is_spam {
            *self.token_spam_counts.get_mut(token).unwrap() += 1;
        } else {
            *self.token_ham_counts.get_mut(token).unwrap() += 1;
        }
    }

    fn incre_msg_cl_count(&mut self, msg: &Message) {
        if msg.is_spam() {
            self.spam_messages_count += 1; 
        } else {
            self.ham_messages_count += 1;
        }
    }

    pub fn predict(&self, text: &str) -> f64 {
        let lower_case_text = text.to_lowercase();
        let tokens = tokenize(&lower_case_text);
        let (prob_if_spam, prob_if_ham): (f64, f64) = self.prob_of_msg(tokens);
        prob_if_spam / (prob_if_spam + prob_if_ham)
    }


    fn prob_of_msg(&self, tokens: HashSet<&str>) -> (f64, f64) {
        let mut log_prob_if_spam = 0.0;
        let mut log_prob_if_ham = 0.0;
        
        for token in self.tokens.iter() {
            let (prob_if_spam, prob_if_ham): (f64, f64) = self.prob_of_tokens(token);
            if tokens.contains(token.as_str()) {
                log_prob_if_spam += prob_if_spam.ln();
                log_prob_if_ham += prob_if_ham.ln();
            } else {
                log_prob_if_spam += (1.0 - prob_if_spam).ln();
                log_prob_if_ham += (1.0 - prob_if_ham).ln();
            }
        }

        (log_prob_if_spam.exp(), log_prob_if_ham.exp())
    }

    fn prob_of_tokens(&self, token: &str) -> (f64, f64) {
        let prob_of_token_spam: f64 = (self.token_spam_counts[token] as f64 + self.alpha)
                                        / (self.spam_messages_count as f64 + 2.0 * self.alpha);

        let prob_of_token_ham: f64 = (self.token_ham_counts[token] as f64 + self.alpha)
                                        / (self.ham_messages_count as f64 + 2.0 * self.alpha);

        (prob_of_token_spam, prob_of_token_ham)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn naive_bayes() {
        let train_messages = [
            Message {
                text: "Free Bitcoin viagra XXX christmas deals 😻😻😻",
                is_spam: true,
            },
            Message::new("My dear Granddaughter, please explain Bitcoin over Christmas dinner"),
    
            Message {
                text: "Here in my garage...",
                is_spam: true,
            },
        ];

        let alpha = 1.;
        let num_spam_messages = 2.;
        let num_ham_messages = 1.;

        let mut model = NativeBayesClassifier::new(alpha);
        model.train(&train_messages);

        let mut expected_tokens: HashSet<String> = HashSet::new();
        for message in train_messages.iter() {
            for token in tokenize(&message.text.to_lowercase()) {
                expected_tokens.insert(token.to_string());
            }
        }

        let input_text = "Bitcoin crypto academy Christmas deals";

        let probs_if_spam = [
            1. - (1. + alpha) / (num_spam_messages + 2. * alpha), // "Free"  (not present)
            (1. + alpha) / (num_spam_messages + 2. * alpha),      // "Bitcoin"  (present)
            1. - (1. + alpha) / (num_spam_messages + 2. * alpha), // "viagra"  (not present)
            1. - (1. + alpha) / (num_spam_messages + 2. * alpha), // "XXX"  (not present)
            (1. + alpha) / (num_spam_messages + 2. * alpha),      // "christmas"  (present)
            (1. + alpha) / (num_spam_messages + 2. * alpha),      // "deals"  (present)
            1. - (1. + alpha) / (num_spam_messages + 2. * alpha), // "my"  (not present)
            1. - (0. + alpha) / (num_spam_messages + 2. * alpha), // "dear"  (not present)
            1. - (0. + alpha) / (num_spam_messages + 2. * alpha), // "granddaughter"  (not present)
            1. - (0. + alpha) / (num_spam_messages + 2. * alpha), // "please"  (not present)
            1. - (0. + alpha) / (num_spam_messages + 2. * alpha), // "explain"  (not present)
            1. - (0. + alpha) / (num_spam_messages + 2. * alpha), // "over"  (not present)
            1. - (0. + alpha) / (num_spam_messages + 2. * alpha), // "dinner"  (not present)
            1. - (1. + alpha) / (num_spam_messages + 2. * alpha), // "here"  (not present)
            1. - (1. + alpha) / (num_spam_messages + 2. * alpha), // "in"  (not present)
            1. - (1. + alpha) / (num_spam_messages + 2. * alpha), // "garage"  (not present)
        ];

        let probs_if_ham = [
            1. - (0. + alpha) / (num_ham_messages + 2. * alpha), // "Free"  (not present)
            (1. + alpha) / (num_ham_messages + 2. * alpha),      // "Bitcoin"  (present)
            1. - (0. + alpha) / (num_ham_messages + 2. * alpha), // "viagra"  (not present)
            1. - (0. + alpha) / (num_ham_messages + 2. * alpha), // "XXX"  (not present)
            (1. + alpha) / (num_ham_messages + 2. * alpha),      // "christmas"  (present)
            (0. + alpha) / (num_ham_messages + 2. * alpha),      // "deals"  (present)
            1. - (1. + alpha) / (num_ham_messages + 2. * alpha), // "my"  (not present)
            1. - (1. + alpha) / (num_ham_messages + 2. * alpha), // "dear"  (not present)
            1. - (1. + alpha) / (num_ham_messages + 2. * alpha), // "granddaughter"  (not present)
            1. - (1. + alpha) / (num_ham_messages + 2. * alpha), // "please"  (not present)
            1. - (1. + alpha) / (num_ham_messages + 2. * alpha), // "explain"  (not present)
            1. - (1. + alpha) / (num_ham_messages + 2. * alpha), // "over"  (not present)
            1. - (1. + alpha) / (num_ham_messages + 2. * alpha), // "dinner"  (not present)
            1. - (0. + alpha) / (num_ham_messages + 2. * alpha), // "here"  (not present)
            1. - (0. + alpha) / (num_ham_messages + 2. * alpha), // "in"  (not present)
            1. - (0. + alpha) / (num_ham_messages + 2. * alpha), // "garage"  (not present)
        ];

        let p_if_spam_log: f64 = probs_if_spam.iter().map(|p| p.ln()).sum();
        let p_if_spam = p_if_spam_log.exp();

        let p_if_ham_log: f64 = probs_if_ham.iter().map(|p| p.ln()).sum();
        let p_if_ham = p_if_ham_log.exp();

        // P(message | spam) / (P(messge | spam) + P(message | ham)) rounds to 0.97
        assert!((model.predict(input_text) - p_if_spam / (p_if_spam + p_if_ham)).abs() < 0.000001);
    }
}