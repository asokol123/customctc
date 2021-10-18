use numpy::PyArray2;
use ordered_float::OrderedFloat;
use pyo3::exceptions::PyAssertionError;
use pyo3::prelude::*;
use pyo3::types::PyString;

#[pymodule]
fn customctc(_py: Python<'_>, _m: &PyModule) -> PyResult<()> {
    #[pyfn(_m)]
    #[pyo3(name = "beam_search")]
    fn beam_search<'py>(
        _py: Python<'py>,
        probs: &PyArray2<f32>,
        alphabet: &PyString,
        beam_size: usize,
    ) -> PyResult<Vec<(f32, String)>> {
        let alphabet = alphabet.to_str()?.as_bytes();
        if probs.shape().len() != 2 {
            return Err(PyAssertionError::new_err(format!(
                "Expected probs to be 2-d, got {}-d",
                probs.shape().len(),
            )));
        }
        let char_length = probs.shape()[0];
        let voc_size = probs.shape()[1];
        if voc_size != alphabet.len() {
            return Err(PyAssertionError::new_err(format!(
                "Expected voc_size ({}) == alphabet size ({})",
                voc_size,
                alphabet.len(),
            )));
        }

        let probs = unsafe { probs.as_array() };
        let mut hypos = probs
            .row(0)
            .as_slice()
            .unwrap()
            .iter()
            .skip(1)
            .enumerate()
            .filter_map(|(i, &p)| {
                if alphabet[i + 1] != b' ' {
                    Some((p, String::from(alphabet[i + 1] as char)))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        for i in 1..char_length {
            let mut heap = std::collections::BTreeSet::<(OrderedFloat<f32>, String)>::new();
            let mut hypos_set = std::collections::HashMap::<String, f32>::new();
            while !hypos.is_empty() {
                let (prob, hypo) = hypos.pop().unwrap();
                for (j, &p) in probs.row(i).as_slice().unwrap()[1..].iter().enumerate() {
                    let cur_prob = prob + p;
                    let cur_hypo = if hypo.as_bytes()[hypo.len() - 1] != alphabet[j + 1] {
                        let mut hypo = hypo.clone();
                        hypo.push(alphabet[j + 1] as char);
                        hypo
                    } else {
                        hypo.clone()
                    };

                    if let Some(&cur_hypo_prob) = hypos_set.get(&cur_hypo) {
                        if cur_hypo_prob < cur_prob {
                            heap.remove(&(OrderedFloat::from(cur_hypo_prob), cur_hypo.clone()));
                            hypos_set.insert(cur_hypo.clone(), cur_prob);
                            heap.insert((OrderedFloat::from(cur_prob), cur_hypo));
                        }
                    } else if heap.len() < beam_size {
                        hypos_set.insert(cur_hypo.clone(), cur_prob);
                        heap.insert((OrderedFloat::from(cur_prob), cur_hypo));
                    } else {
                        let first = heap.iter().next().unwrap();
                        if first.0.into_inner() < cur_prob {
                            let first = first.clone();
                            heap.remove(&first);
                            hypos_set.remove(&first.1);
                            hypos_set.insert(cur_hypo.clone(), cur_prob);
                            heap.insert((OrderedFloat::from(cur_prob), cur_hypo));
                        }
                    }
                }
            }

            hypos = heap
                .into_iter()
                .map(|(prob, hypo)| (prob.into_inner(), hypo))
                .collect();
        }

        Ok(hypos)
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
