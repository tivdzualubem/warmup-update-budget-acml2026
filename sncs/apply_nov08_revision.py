"""Add the verified November execution evidence; preserve original selections."""
from pathlib import Path
import json
from apply_oct29_revision import build as previous_build
ROOT=Path(__file__).resolve().parent
M=ROOT/'manuscript'

def build():
    previous_build()
    audit=json.loads((ROOT/'evidence/execution_provenance/execution_provenance_summary.json').read_text())
    assert audit['status']=='passed' and not audit['failures']
    main=(M/'Research_Manuscript.md').read_text(encoding='utf-8')
    supp=(M/'All_Seed_Results_and_Provenance.md').read_text(encoding='utf-8')
    title=main.splitlines()[0]
    main=main.replace('extends training to 800 epochs.','extends training to 800 epochs. The amplified-update intervention spans the first five epochs, rather than one transition.')
    main=main.replace('These studies motivate controlling early updates;',
        'Liu et al. (2020) also analyze early adaptive-rate variance. These studies motivate controlling early updates;')
    main=main.replace('Scratch runs use AG News and DBpedia,',
        'The Transformer architecture follows Vaswani et al. (2017). Huang et al. (2020) and Xiong et al. (2020) study initialization and normalization, which are held fixed within each paired comparison here. Scratch runs use AG News and DBpedia,')
    main=main.replace('trained for three epochs with AdamW and 10% linear warmup.',
        'trained for three epochs with AdamW (Loshchilov and Hutter, 2019) and 10% linear warmup.')
    provenance='''### 3.9 Execution evidence and reproducibility

The returned master manifest and four dataset manifests retain negative completion flags. All five are byte-identical to their pre-execution launcher counterparts, as are all 49 returned design/configuration/environment JSON files. They describe the packaged starting state and were not updated to reflect execution. We preserve those flags and establish recorded outcomes separately from per-job status, manifest, history and trace files. These records cover the expected 240-job matrix: 180 trained test endpoints and 60 diagnostic records. Twenty additional first-transition severity records are separate.

The audit checks 1,573 history rows and 55,535 recorded transition rows. All 277,675 stored LR fields agree with independently reconstructed condition-specific schedules. All seven initial-state/data/gradient fingerprints agree within each of 40 six-condition groups. The scientific source/configuration matches the original launcher in 224 file copies (56 distinct files across four datasets). These agreements support source-compatible recorded execution; they are not an execution-time attestation of every source byte or a rerun of the original training.

A separately returned AG News console log contains 40 scratch-job completions and 635 epoch rows; every printed loss and step agrees with the corresponding returned history at its four-decimal precision. The same log records interrupted launches and stops before the remaining phases. Complete final orchestration logs and the final executed launcher patches have not been recovered. Saved split hashes identify recorded partitions but do not recover the original index arrays or token-to-ID mapping. Candidate-epoch weights and prediction-level validation evidence also remain unavailable. The supplement distinguishes these limits from the reproducible saved-record analysis.

A minimal CPU example imports the archived scheduling policy and Adam constructor, executes the first two transitions, and observes the LR at each actual optimizer call. It distinguishes rejected candidates from committed replays and confirms that a zero-LR transition can change Adam moments and its counter without changing weights. It is a fixed-gradient fixture, not a research training run or certification of the original device.

'''
    provenance=provenance.replace('Complete final orchestration logs and the final executed launcher patches have not been recovered.',
        'The recovered 11 October hotfix addresses both observed launch errors without changing scientific modules. Its application on the original host and the final executed wrapper bytes are not independently attested by a complete final log.')
    main=main.replace('## 4. Scratch results under the recorded procedures',provenance+'## 4. Scratch results under the recorded procedures')
    main=main.replace('If practical relevance requires another experiment, a targeted pair could consume',
        'The next prioritized test is a small matched pair that would consume')
    main=main.replace('This is a proposal, not a reported experiment; the extreme-fault recovery extension addresses a different question.',
        'That comparison has not been run. The present evidence therefore does not establish downstream harm from accidentally consuming the configured peak. The extreme-fault recovery extension addresses a different question and remains deferred.')
    supp+='''

## Execution provenance: design metadata and returned job outcomes

Five negative completion flags in the master/dataset manifests are unchanged from the original pre-execution launcher. All 49 non-result JSON files in the returned package match the source launcher byte for byte. The flags are retained as historical metadata; job execution is evaluated from the accompanying result, status, job-manifest, history and transition records. Their identity and file digests are exported per run in execution_provenance_per_run.csv.

| Cohort | Recorded termination | Records | Trained test endpoints |
|:--|:--|--:|--:|
| Scratch | Early stopping | 89 | 89 |
| Scratch | Maximum epochs | 11 | 11 |
| Scratch | Guard fail-fast | 20 | 0 |
| Pretrained | Three epochs, ordinary training arms | 80 | 80 |
| Pretrained | Three epochs, WNUT fail-fast arms whose guard did not trigger | 5 | 0 |
| Pretrained | Guard fail-fast | 15 | 0 |
| Pretrained | LR assertion before mutation | 20 | 0 |

Thus 240 scientific records contain 180 trained test endpoints. The five WNUT diagnostic arms do not become additional test classifiers merely because their guard missed and training continued. Twenty separate pretrained severity records are not added to this matrix.

The source comparison covers 224 byte-identical scientific Python/configuration copies in the original launcher, representing 56 distinct files across four dataset copies. All 1,573 saved history rows and 55,535 trace rows are checked against their job records; 277,675 LR fields match independently reconstructed schedules. All seven fingerprint fields match in each of 40 initialization groups. The two scratch dataset split manifests are identical across their respective 60 jobs. These checks establish consistency of saved evidence with the preserved implementation. They do not prove the final host executed unmodified source, reconstruct unavailable model states, or establish an exact training rerun.

The separate AG News log has 1,499 lines. Its 635 rounded epoch entries and 40 completion reasons agree with the returned scratch histories. Two queue-handling failures precede successful phases; a later cross-dataset threshold lookup error interrupts the guarded-phase launch. The log is therefore partial. The returned records include later outcomes, but the final launcher/hotfix bytes, complete master status and console logs for other phases are unavailable in the examined package. The original environment files are design records, not an independently captured final host dependency lock.

The original scratch runner saves a rolling epoch-end resume state and overwrites its validation-best state. It deletes both temporary files after normal completion. This retention policy is consistent with their absence from returned results; it does not establish whether manual backups survive. No eligible candidate-epoch weights, matching validation predictions or full paired states immediately after transition 49,999 were recovered. The older large checkpoint folder mentioned in correspondence belongs to an April three-seed DBpedia cohort, not this October study.

## Executable first-two-step and zero-LR examples

The supplied example uses three float64 CPU parameters and two fixed clipped gradients. It imports the frozen scratch runner's condition policy, model-specific schedule and Adam constructor, and instruments actual optimizer calls. It has no dataset input. The threshold used to illustrate rescue is a fixture value, not a new research calibration.

| Condition | Actual LR calls at transition 1 | Actual LR calls at transition 2 |
|:--|:--|:--|
| Stable | S(1) | S(2) |
| Single bad then stable | 1.0 | S(2) |
| Corrected legacy | L(1) | L(2) |
| Late legacy | 1.0 | L(1) |
| Legacy rescue | 1.0 rejected, then L(1) replayed | L(1) |
| Archived fail-fast | 1.0, retained before stop | No transition |
| LR assertion | No optimizer call | No transition |

For widths 128 and 256, the example observes both the candidate and replay calls. The replay restores the initial Adam counter before updating, so two calls at the first transition leave one committed step. Its retained state matches a clean legacy first transition. At transition two the rescued arm still consumes L(1); correction of the first transition did not remove later lag. The scratch isolated-fault arm exits transition one with stored LR=1.0, then installs S(2) before its next update.

The assertion rejects before mutation. Its archived result field named candidate_lr holds the entry value even though no optimizer call consumed that value; the example reports no actual consumed LR. A separate three-update fixture installs S(50,000)=0 on its last call. Parameters remain unchanged while both Adam moment arrays and the counter change. This is a boundary-value fixture, not a recovered or newly trained 49,999-step state.

## Reproduction scope

The exact previously delivered package was extracted into a fresh directory and its documented command rerun. All 44 existing tests passed; scientific CSVs, figure images and both manuscript texts matched the delivered copies, and PDF page counts and extracted content matched. This closes a concrete current reproduction check. No specific earlier discrepant output was supplied, so it does not retrospectively identify the cause of an unspecified prior discrepancy.

The updated package adds the record-provenance audit and actual optimizer-call example. Its execution logs, commands, dependency versions and file digests accompany the sources. Saved-record analysis, CPU fixtures and typesetting are reproducible separately from original model training. No new research model training, corrected checkpoint reselection or prediction-level reevaluation is reported. The configured-peak pair is prospective; full extreme-fault recovery remains unperformed.
'''
    supp=supp.replace('The returned records include later outcomes, but the final launcher/hotfix bytes, complete master status and console logs for other phases are unavailable in the examined package.',
        'The recovered 11 October hotfix unwraps queued job tuples and scopes the matrix enumerators to the current dataset before threshold lookup. Applying its pure text transformation to the original four launchers in memory produces syntactically valid, idempotent corrections without modifying scientific modules or results. These reconstructed wrappers explain the supplied repair but are not recovered execution-time files. Final patch-application logs, complete master status and console logs for other phases remain unavailable.')
    supp=supp.replace(' The older large checkpoint folder mentioned in correspondence belongs to an April three-seed DBpedia cohort, not this October study.','')
    start=supp.index('The exact previously delivered package was extracted into a fresh directory')
    end=supp.index('The updated package adds the record-provenance audit',start)
    supp=supp[:start]+'''The supplied analyses reconstruct tables and figures from the preserved result archive and historical notebook exports. The reproducibility command also checks unchanged source/data digests, original scientific CSVs, displayed results, shared-step arithmetic, evaluator behavior and rollback fixtures. Original inputs are preserved separately from derived outputs.

'''+supp[end:]
    assert main.splitlines()[0]==title
    (M/'Research_Manuscript.md').write_text(main,encoding='utf-8')
    (M/'All_Seed_Results_and_Provenance.md').write_text(supp,encoding='utf-8')

if __name__=='__main__':build()
