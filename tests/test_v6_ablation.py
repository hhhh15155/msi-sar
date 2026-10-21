"""Approved V6 ablations: common initialization and active forward paths."""
from pathlib import Path
import json
import tempfile
import unittest

import torch

from models.fuzzy_transformer import FuzzyTransformer
from models.main_classification import build_main_model


class V6AblationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def build(self, variant, ms=6, sar=1):
        return build_main_model({'variant':variant,'ms_channels':ms,'sar_channels':sar},5)

    def test_default_v6_matches_unmodified_class_exactly(self):
        torch.manual_seed(202201)
        original=FuzzyTransformer(6,1,5).eval()
        torch.manual_seed(202201)
        current=self.build('v6_current').eval()
        for name,value in original.state_dict().items():
            torch.testing.assert_close(value,current.state_dict()[name],rtol=0,atol=0)
        ms,sar=torch.randn(2,6,5,5),torch.randn(2,1,5,5)
        with torch.inference_mode():
            torch.testing.assert_close(original(ms,sar)['logits'],current(ms,sar)['logits'],rtol=0,atol=0)

    def test_both_ablation_common_initial_weights_are_identical_to_full_v6(self):
        torch.manual_seed(202201)
        reference=self.build('v6_current').state_dict()
        for variant in ('a01_late_equal','a02_no_early_path'):
            torch.manual_seed(202201)
            model=self.build(variant)
            self.assertEqual(model.ms_stem.channel_mix.out_channels,192)
            self.assertEqual(model.auxiliary_weight,0.2)
            for name,value in model.state_dict().items():
                torch.testing.assert_close(value,reference[name],rtol=0,atol=0)
            if variant=='a01_late_equal':
                self.assertEqual(model.stages[1]['fusion'].mode,'concat')
                self.assertFalse(hasattr(model.stages[1]['fusion'],'memberships'))
                self.assertEqual(model.stages[0]['fusion'].mode,'fuzzy')
            else:
                self.assertNotIn('fusion',model.stages[0])
                self.assertFalse(hasattr(model,'ms_exchange'))
                self.assertFalse(hasattr(model,'sar_exchange'))
                self.assertEqual(model.stages[1]['fusion'].mode,'fuzzy')

    def test_two_channel_configs_odd_patches_gradients_and_loss(self):
        for variant in ('a01_late_equal','a02_no_early_path'):
            for ms_c,sar_c,patch in ((6,1,5),(10,4,11)):
                model=self.build(variant,ms_c,sar_c)
                before=set(dict(model.named_parameters()))
                result=model(torch.randn(2,ms_c,patch,patch),torch.randn(2,sar_c,patch,patch),
                             labels=torch.tensor([1,3]),return_fusion_weights=True)
                self.assertEqual(result['logits'].shape,(2,5))
                losses=result['losses']
                torch.testing.assert_close(losses['total'],losses['ce']+0.2*losses['auxiliary'])
                losses['total'].backward()
                self.assertEqual(before,set(dict(model.named_parameters())))
                self.assertEqual([n for n,p in model.named_parameters() if p.grad is None],[])
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
                self.assertEqual(len(result['fusion_weights']),2)
                missing=1 if variant=='a01_late_equal' else 0
                self.assertIsNone(result['fusion_weights'][missing])

    def test_no_early_path_keeps_all_four_encoders_and_final_fusion(self):
        model=self.build('a02_no_early_path').eval()
        called=[]
        handles=[]
        for i,stage in enumerate(model.stages):
            for name in ('ms','sar'):
                handles.append(stage[name].register_forward_hook(lambda m,a,o,key=(i,name):called.append(key)))
        handles.append(model.stages[1]['fusion'].register_forward_hook(lambda m,a,o:called.append('last_fusion')))
        ms,sar=torch.randn(2,6,5,5),torch.randn(2,1,5,5)
        rng=torch.get_rng_state().clone()
        with torch.inference_mode():
            first=model(ms,sar)['logits'];second=model(ms,sar)['logits']
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        torch.testing.assert_close(first,second,rtol=0,atol=0)
        self.assertEqual(called,[(0,'ms'),(0,'sar'),(1,'ms'),(1,'sar'),'last_fusion']*2)
        for handle in handles:handle.remove()

    def test_only_six_unfinished_ablations_keep_original_protocol(self):
        import shutil,yaml
        from scripts.generate_v6_ablation_configs import generate
        from scripts import main_experiment as engine
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            shutil.copytree(engine.ROOT/'configs/main_first_pass',root/'configs/main_first_pass')
            ablations=generate(root)
            self.assertEqual(len(ablations['experiments']),6)
            self.assertEqual(set(ablations['variants']),{'a01_late_equal','a02_no_early_path'})
            self.assertEqual(set(ablations['datasets']),{'yrd','yrd2509new','grss07'})
            self.assertEqual(ablations['screen_session'],'msi_v6_ablate_gpu0')
            self.assertFalse((root/'configs/v6_ablation/controls_manifest.json').exists())
            for job in ablations['experiments']:
                config=yaml.safe_load((root/job['config']).read_text())
                old=yaml.safe_load((root/'configs/main_first_pass'/('v6_current_'+job['dataset']+'.yaml')).read_text())
                self.assertEqual({k:v for k,v in config.items() if k!='variant'},
                                 {k:v for k,v in old.items() if k!='variant'})

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA deployment check')
    def test_gpu_fp32_training_and_test_batch_1024(self):
        for variant in ('a01_late_equal','a02_no_early_path'):
            for ms_c,sar_c in ((6,1),(10,4)):
                model=self.build(variant,ms_c,sar_c).cuda()
                loss=model(torch.randn(4,ms_c,11,11,device='cuda'),
                           torch.randn(4,sar_c,11,11,device='cuda'),
                           labels=torch.tensor([0,1,2,3],device='cuda'))['losses']['total']
                loss.backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                    for p in model.parameters()))
                model.zero_grad(set_to_none=True)
                model.eval()
                torch.cuda.reset_peak_memory_stats()
                with torch.inference_mode():
                    out=model(torch.randn(1024,ms_c,11,11,device='cuda'),
                              torch.randn(1024,sar_c,11,11,device='cuda'))['logits']
                self.assertEqual(out.shape,(1024,5))
                self.assertTrue(torch.isfinite(out).all())
                print('V6 GPU CHECK',variant,ms_c,sar_c,'parameters=',sum(p.numel() for p in model.parameters()),
                      'peak_MiB=',round(torch.cuda.max_memory_allocated()/1024**2,1),flush=True)
                del model,out,loss
                torch.cuda.empty_cache()

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA resume integration check')
    def test_resume_matches_uninterrupted_predictions_and_logits(self):
        import numpy as np,yaml
        from unittest.mock import patch
        from scripts import main_experiment as engine
        (engine.ROOT/'tmp').mkdir(exist_ok=True)
        for variant in ('a01_late_equal','a02_no_early_path'):
            with tempfile.TemporaryDirectory(dir=engine.ROOT/'tmp') as temporary:
                root=Path(temporary)
                prepared=root/'prepared/synthetic'
                prepared.mkdir(parents=True)
                image=np.random.default_rng(2).normal(size=(7,8,7)).astype(np.float32)
                gt=(np.arange(56).reshape(7,8)%2).astype(np.int64)
                train,test=np.arange(8),np.arange(8,56)
                np.save(prepared/'image.npy',image)
                np.save(prepared/'gt.npy',gt)
                np.savez(prepared/'split.npz',train_indices=train,test_indices=test)
                engine.atomic_json(prepared/'metadata.json',{'class_names':['a','b'],
                    'train_counts':[4,4],'test_counts':[24,24],
                    'prepared_sha256':{'split.npz':engine.digest(prepared/'split.npz')}})
                config={'variant':variant,'ms_channels':6,'sar_channels':1,'seed':202201,
                    'patch_size':5,'epochs':2,'batch_size':4,'test_batch_size':16,
                    'learning_rate':0.0005,'weight_decay':0.0001,'precision':'fp32'}
                path=root/'config.yaml'
                path.write_text(yaml.safe_dump(config))
                manifest={'output_root':str(root),'experiments':[
                    {'id':n,'dataset':'synthetic','config':str(path)} for n in ('full','resume')]}
                engine.train(manifest,'full')
                original=engine.save_checkpoint
                def interrupt(path,payload):
                    original(path,payload)
                    if path.name=='resume.pth' and payload['epoch']==1:
                        raise RuntimeError('v6 interruption')
                with patch.object(engine,'save_checkpoint',side_effect=interrupt):
                    with self.assertRaisesRegex(RuntimeError,'v6 interruption'):
                        engine.train(manifest,'resume')
                engine.train(manifest,'resume')
                np.testing.assert_array_equal(np.load(root/'full/test_predictions.npy'),
                                              np.load(root/'resume/test_predictions.npy'))
                left,right=build_main_model(config,2).eval(),build_main_model(config,2).eval()
                left.load_state_dict(torch.load(root/'full/model_last.pth',map_location='cpu',weights_only=True))
                right.load_state_dict(torch.load(root/'resume/model_last.pth',map_location='cpu',weights_only=True))
                inputs=engine.PatchBank(image,5,torch.device('cpu')).take(torch.tensor(test))
                with torch.inference_mode():
                    torch.testing.assert_close(left(inputs[:,:6],inputs[:,6:])['logits'],
                                               right(inputs[:,:6],inputs[:,6:])['logits'],rtol=1e-5,atol=1e-5)


if __name__=='__main__':unittest.main()
