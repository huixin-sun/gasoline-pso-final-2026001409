"""Extract the fixed feature configuration by instrument tag, avoiding ambiguous offsets."""
import argparse,csv,json,hashlib
from pathlib import Path
import openpyxl

TAGS=['S-ZORB.TC_2801.PV','S-ZORB.FT_1001.PV','S-ZORB.FT_9401.PV',
 'S-ZORB.FT_5104.PV','S-ZORB.PT_7103B.DACA','S-ZORB.FT_9201.PV',
 'S-ZORB.FC_5001.DACA','S-ZORB.FC_1203.PV','S-ZORB.FC_2702.DACA',
 'S-ZORB.PDT_1003.DACA','S-ZORB.FT_1301.DACA','S-ZORB.FC_3101.PV',
 'S-ZORB.TE_5001.DACA','S-ZORB.SIS_PT_2602.PV','S-ZORB.TE_5006.DACA',
 'S-ZORB.FT_2001.DACA','S-ZORB.BS_AT_2402.PV','S-ZORB.FT_9102.PV',
 'S-ZORB.LT_9101.DACA','S-ZORB.TXE_3201A.DACA','S-ZORB.TE_9002.DACA',
 'S-ZORB.PT_1102.DACA','S-ZORB.PDT_2606.DACA']

def main():
 p=argparse.ArgumentParser();p.add_argument('--samples',required=True);p.add_argument('--variables',required=True);args=p.parse_args()
 root=Path(__file__).resolve().parents[1]; dest=root/'data';dest.mkdir(exist_ok=True)
 rows=list(openpyxl.load_workbook(args.samples,read_only=True,data_only=True).active.values)
 info=list(openpyxl.load_workbook(args.variables,read_only=True,data_only=True).active.values)
 by_tag={str(r[1]).strip():r for r in info[1:] if r[1]}
 cols={str(v).strip():i for i,v in enumerate(rows[1]) if v}
 fixed=['feed_sulfur','feed_ron','feed_saturates','feed_density']
 indices=[2,3,4,8]+[cols[t] for t in TAGS]
 headers=['sample_id','time']+fixed+TAGS+['product_ron','product_sulfur']
 records=[]
 for r in rows[3:]:
  if not isinstance(r[0],(int,float)):continue
  values=[r[i] for i in indices]+[r[10],r[9]]
  records.append([int(r[0]),str(r[1]).strip()]+values)
 assert len(records)==325 and all(v is not None for r in records for v in r)
 with (dest/'samples.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.writer(f);w.writerow(headers);w.writerows(records)
 # Parentheses and negative endpoints in the supplied range strings are not parsed heuristically.
 # These manually transcribed numeric endpoints are checked against original_range in the dictionary.
 bounds=[(200,350),(40,145),(25,900),(0,3500),(0,2.5),(250,900),(500,3800),(5,15),(15,50),(-1.5,12),
 (100,3500),(90,550),(50,150),(-120,150),(100,150),(-8,75),(-.5,5),(20,20000000),(-5,35),(300,550),(10,50),(2.55,2.95),(-.5,20)]
 dictionary=[]
 for t,(lo,hi) in zip(TAGS,bounds):
  r=by_tag[t];dictionary.append({'tag':t,'name':r[2],'lower':lo,'upper':hi,'unit':r[4],'delta':r[5],
   'original_range':r[3],'original_variable_id':r[0],'excel_column_1based':cols[t]+1})
 (dest/'variables.json').write_text(json.dumps(dictionary,ensure_ascii=False,indent=2),encoding='utf-8')
 sources=[{'name':Path(a).name,'sha256':hashlib.sha256(Path(a).read_bytes()).hexdigest()} for a in [args.samples,args.variables]]
 (dest/'provenance.json').write_text(json.dumps({'sources':sources,'rows':325,'fixed_columns':fixed,'selected_controls':23,
 'selection':'Fixed feature configuration; methodological attribution is listed in the project references. EH-103 uses TXE_3201A with range 300-550. No feature selection on holdout. All control bounds follow the attachment by instrument tag.',
 'extraction':'Original Sheet1 rows 4:328. Product RON column K; sulfur J. Four fixed features C,D,E,I.'},ensure_ascii=False,indent=2),encoding='utf-8')
 print('Extracted 325 complete rows, 4 fixed features, 23 controls, 2 targets.')

if __name__=='__main__':main()
