import copy
import importlib.util
import json
import tempfile
import unittest
import zipfile
from html.parser import HTMLParser
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder=load('builder',ROOT/'scripts/build_report.py')
packager=load('packager',ROOT/'scripts/package_skill.py')


class HTMLAudit(HTMLParser):
    def __init__(self):
        super().__init__();self.ids=[];self.anchors=[];self.remote_resources=[];self.scripts=0
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.append(a['id'])
        if a.get('href','').startswith('#'):self.anchors.append(a['href'][1:])
        if tag=='script':self.scripts+=1
        for key in ('src',):
            if a.get(key,'').startswith(('https:','http:','//')):self.remote_resources.append(a[key])
        if tag=='link' and a.get('href','').startswith(('https:','http:','//')):self.remote_resources.append(a['href'])


class ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=json.loads((ROOT/'examples/nanogpt/report-data.json').read_text(encoding='utf-8'))

    def test_demo_valid_and_offline_navigation_complete(self):
        self.assertEqual(builder.validate(self.data),[])
        audit=HTMLAudit();audit.feed(builder.render(self.data))
        self.assertEqual(len(audit.ids),len(set(audit.ids)))
        self.assertFalse(set(audit.anchors)-set(audit.ids))
        self.assertEqual(audit.remote_resources,[])

    def test_every_core_branch_has_example_path(self):
        for flow in self.data['flows']:
            seen_nodes=set();seen_edges=set()
            for case in self.data['examples']:
                ids=[v['node'] for v in case['paths'].get(flow['id'],[])]
                seen_nodes.update(ids);seen_edges.update(zip(ids,ids[1:]))
            self.assertEqual(seen_nodes,{n['id'] for n in flow['nodes']})
            self.assertEqual(seen_edges,{(e['from'],e['to']) for e in flow['edges']})

    def test_reject_dangling_edges(self):
        data=copy.deepcopy(self.data);data['flows'][0]['edges'][0]['to']='does-not-exist'
        self.assertTrue(builder.validate(data))

    def test_reject_decision_with_missing_exit(self):
        data=copy.deepcopy(self.data)
        data['flows'][0]['edges']=[e for e in data['flows'][0]['edges'] if not(e['from']=='F02' and e['to']=='FX')]
        self.assertTrue(any('two labeled exits' in e for e in builder.validate(data)))

    def test_reject_unsupported_trace_jump(self):
        data=copy.deepcopy(self.data)
        data['examples'][0]['paths']['forward'].insert(1,dict(node='F09',input='a',output='b'))
        self.assertTrue(any('nonexistent edge' in e for e in builder.validate(data)))

    def test_reject_duplicate_ids_and_unknown_evidence(self):
        data=copy.deepcopy(self.data);data['flows'][0]['nodes'][0]['sources']=['missing']
        self.assertTrue(any('unknown source' in e for e in builder.validate(data)))
        data=copy.deepcopy(self.data);data['flows'][1]['id']=data['flows'][0]['id']
        self.assertTrue(any('duplicate flow' in e for e in builder.validate(data)))

    def test_source_and_json_script_injection_are_text(self):
        data=copy.deepcopy(self.data)
        attack='</script><script>globalThis.INJECTED=true</script><img src=x onerror=alert(1)>'
        data['examples'][0]['input']=attack;data['sources'][0]['code']=attack
        output=builder.render(data)
        audit=HTMLAudit();audit.feed(output)
        self.assertEqual(audit.scripts,2)
        self.assertNotIn(attack,output)
        self.assertIn('&lt;img',output)
        self.assertIn('\\u003c/script\\u003e',output)

    def test_source_ranges_match_snapshot(self):
        for s in self.data['sources']:
            file=s['url'].split('/')[-1].split('#')[0]
            original=(ROOT/'examples/nanogpt/source'/file).read_text(encoding='utf-8-sig').splitlines()
            for line in s['code'].splitlines():
                # Source prefix has fixed minimum width, not a separator search.
                number=int(line[:3]);text=line[5:]
                self.assertEqual(text,original[number-1])

    def test_install_zip_is_minimal_and_licensed(self):
        scratch=(ROOT/'work').resolve()
        self.assertTrue(scratch.is_relative_to(ROOT.resolve()))
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            target=packager.package(Path(directory)/'skill.zip')
            with zipfile.ZipFile(target) as archive:
                files=set(archive.namelist())
                self.assertIsNone(archive.testzip())
                self.assertIn('SKILL.md',files);self.assertIn('LICENSE',files)
                self.assertIn('scripts/build_report.py',files)
                self.assertFalse(any(p.startswith(('examples/','tests/')) or '__pycache__' in p for p in files))


if __name__=='__main__':
    unittest.main()
