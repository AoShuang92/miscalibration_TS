# miscalibration_TS
The paper has been accepted in the UAI 2023, with the title "Two Sides of Miscalibration: Identifying Over and Under-Confidence Prediction for Network Calibration". <br>

Proper confidence calibration of deep neural networks is essential in safety-critical tasks for reliable predictions. Miscalibration can lead to model over-confidence and/or under-confidence; i.e., the predicted confidence can be greater or less than model accuracy. Recent studies have highlighted the over-confidence issue by introducing calibration techniques and demonstrated success on various tasks. However, miscalibration through under-confidence has not yet to receive much attention. In this paper, we address the necessity of paying attention to the under-confidence issue. We first introduce a novel metric, a miscalibration score, to identify the overall and class-wise calibration status, including being over or under-confident. Our proposed metric reveals the pitfalls of existing calibration techniques, where they often overly calibrate the model and worsen under-confident predictions. Then we utilize the class-wise miscalibration score as a proxy to design a calibration technique that can tackle both over and under-confidence. We report extensive experiments that show that our proposed methods substantially outperform existing calibration techniques. We also validate the automatic failure detection of the proposed calibration technique using our class-wise miscalibration score with a risk-coverage curve. Results show that our methods significantly improve failure detection as well as trustworthiness of the model. 


## Reference 
If you use the code, please cite the paper as below:
```
@inproceedings{ao2023two,
  title={Two sides of miscalibration: identifying over and under-confidence prediction for network calibration},
  author={Ao, Shuang and Rueger, Stefan and Siddharthan, Advaith},
  booktitle={Uncertainty in Artificial Intelligence},
  pages={77--87},
  year={2023},
  organization={PMLR}
}
```
