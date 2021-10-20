use numpy::PyArray2;
use ordered_float::OrderedFloat;
use pyo3::exceptions::PyAssertionError;
use pyo3::prelude::*;
use pyo3::types::{PyFloat, PyString};

#[pymodule]
fn customctc(_py: Python<'_>, _m: &PyModule) -> PyResult<()> {
    #[pyfn(_m, beam_size = 100)]
    #[pyo3(name = "beam_search", text_signature = "(probs, alphabet, beam_size = 100, lm = None")]
    fn beam_search<'py>(
        _py: Python<'py>,
        probs: &PyArray2<f32>,
        alphabet: &PyString,
        beam_size: usize,
        lm: Option<&PyAny>,
    ) -> PyResult<Vec<(f32, String, u8)>> {
        let alphabet = alphabet.to_str()?.as_bytes();
        if probs.shape().len() != 2 {
            return Err(PyAssertionError::new_err(format!(
                "Expected probs to be 2-d, got {}-d",
                probs.shape().len(),
            )));
        }
        let voc_size = probs.shape()[1];
        if voc_size != alphabet.len() {
            return Err(PyAssertionError::new_err(format!(
                "Expected voc_size ({}) == alphabet size ({})",
                voc_size,
                alphabet.len(),
            )));
        }

        let probs = unsafe { probs.as_array() };

        let mut hypos = vec![(1., String::new(), b'^')];
        for i in 0..probs.shape()[0] {
            let mut heap = std::collections::BTreeSet::<(OrderedFloat<f32>, String, u8)>::new();
            let mut hypos_set = std::collections::HashMap::new();
            for (prob, hypo, path) in hypos.drain(..) {
                for (j, &p) in probs.row(i).iter().enumerate() {
                    let cur_path = alphabet[j];
                    let mut cur_prob = prob * p;
                    let cur_hypo = if path == cur_path || j == 0 {
                        hypo.clone()
                    } else {
                        let mut hypo = hypo.clone();
                        hypo.push(cur_path as char);
                        hypo
                    };

                    if let Some(lm) = lm {
                        let lm_prob: f64 = lm
                            .call_method1("score", (&cur_hypo,))?
                            .downcast::<PyFloat>()?
                            .value();

                        cur_prob += (0.9 * lm_prob.exp() + 0.0001 * (i as f64)) as f32;
                    }

                    if let Some(&prev_prob) = hypos_set.get(&(cur_hypo.clone(), cur_path)) {
                        heap.remove(&(OrderedFloat::from(prev_prob), cur_hypo.clone(), cur_path));
                        heap.insert((
                            OrderedFloat::from(cur_prob + prev_prob),
                            cur_hypo.clone(),
                            cur_path,
                        ));
                        hypos_set.insert((cur_hypo, cur_path), cur_prob + prev_prob);
                    } else if heap.len() < beam_size {
                        heap.insert((OrderedFloat::from(cur_prob), cur_hypo.clone(), cur_path));
                        hypos_set.insert((cur_hypo, cur_path), cur_prob);
                    } else {
                        let first = heap.iter().next().unwrap();
                        let first_prob = first.0.into_inner();
                        if first_prob < cur_prob {
                            let first = first.clone();
                            heap.remove(&first);
                            hypos_set.remove(&(first.1.clone(), first.2));
                            heap.insert((OrderedFloat::from(cur_prob), cur_hypo.clone(), cur_path));
                            hypos_set.insert((cur_hypo, cur_path), cur_prob);
                        }
                    }
                }
            }
            hypos = heap
                .into_iter()
                .map(|(prob, hypo, path)| (prob.into_inner(), hypo, path))
                .collect();
        }

        Ok(hypos)
    }

    Ok(())
}
