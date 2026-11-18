"""Versioned scratch-classification evaluator for future experiment integration."""
import numpy as np
import torch
from .evaluation import CrossEntropyAccumulator,EVALUATOR_VERSION
from .resumable_state import preserve_rng

def evaluate(model,loader,device,num_classes):
    """Evaluate example-weighted CE/NLL; preserve model mode and training RNG.

    Evaluation loaders must use the intended fixed evaluation population. This
    utility does not restore arbitrary custom sampler/augmentation side effects.
    """
    previous_mode=model.training
    generator=getattr(loader,'generator',None)
    generator_state=generator.get_state() if generator is not None else None
    accumulator=CrossEntropyAccumulator();probs_all=[];labels_all=[]
    try:
        with preserve_rng(),torch.no_grad():
            model.eval()
            for batch in loader:
                labels=batch['labels'].to(device)
                logits=model(batch['input_ids'].to(device))
                accumulator.update(logits,labels)
                probs_all.append(torch.softmax(logits.to(torch.float64),dim=1).cpu().numpy())
                labels_all.append(labels.cpu().numpy())
    finally:
        model.train(previous_mode)
        if generator is not None:generator.set_state(generator_state)
    loss=accumulator.mean
    probs=np.concatenate(probs_all);labels=np.concatenate(labels_all);pred=probs.argmax(axis=1)
    f1=[]
    for k in range(num_classes):
        tp=np.sum((pred==k)&(labels==k));fp=np.sum((pred==k)&(labels!=k));fn=np.sum((pred!=k)&(labels==k))
        f1.append(float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.)
    confidence=probs.max(axis=1);correct=pred==labels;ece=0.
    for low,high in zip(np.linspace(0,1,16)[:-1],np.linspace(0,1,16)[1:]):
        mask=(confidence>low)&(confidence<=high)
        if mask.any():ece+=float(mask.mean()*abs(correct[mask].mean()-confidence[mask].mean()))
    target=np.eye(num_classes)[labels]
    return dict(loss=loss,nll=loss,accuracy=float(correct.mean()),macro_f1=float(np.mean(f1)),
                ece=ece,brier=float(np.mean(np.sum((probs-target)**2,axis=1))),
                num_examples=accumulator.num_examples,evaluator_version=EVALUATOR_VERSION,
                loss_definition='sample_weighted_logit_cross_entropy_no_probability_cap')
