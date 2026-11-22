"""Render the verified seven-role brief as a standalone, printable HTML file."""
from html import escape
from pathlib import Path
from comparison_brief import briefs
ROOT=Path(__file__).resolve().parent
OUT=ROOT.parents[1]/"outputs/judgment-role/experiment-results.html"
CSS="""
:root{color-scheme:light;--ink:#1a2c32;--teal:#066456;--line:#d5deda;--pale:#ecf3ef}
*{box-sizing:border-box}body{margin:0;color:var(--ink);background:#fafbf8;font:16px/1.55 system-ui,sans-serif}
main{max-width:1010px;margin:auto;background:#fff;padding:38px 5vw 70px}header{display:flex;align-items:center;justify-content:space-between;border-bottom:2px solid var(--teal);color:var(--teal);padding:0 0 12px}header strong{font-size:1.3rem}
h1,h2{font-family:Georgia,serif;font-weight:400;line-height:1.12}h1{font-size:clamp(2rem,5vw,3.2rem);margin:30px 0 12px}h2{font-size:2rem;color:var(--ink);margin:0 0 12px}h3{color:var(--teal);font-size:1.15rem;margin:20px 0 4px}.sub{font-size:1.1rem;color:#456068}p{max-width:79ch;margin:.7em 0}
section{border-bottom:1px solid var(--line);padding:32px 0}section:first-of-type{padding-top:12px}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;margin:14px 0;font-size:.94rem}th,td{border-bottom:1px solid var(--line);text-align:left;vertical-align:top;padding:9px 11px}th{background:var(--pale);color:var(--teal)}
a{color:var(--teal);text-underline-offset:3px}.sources{display:flex;column-gap:16px;row-gap:4px;flex-wrap:wrap;font-size:.87rem;margin-top:13px}
nav{display:flex;flex-wrap:wrap;gap:6px 18px;margin-top:16px}nav a{font-size:.9rem}
@media(max-width:620px){main{padding:20px}th,td{padding:7px;font-size:.82rem}header span{font-size:.8rem}}
@media print{main{padding:0;max-width:none}header,nav{display:none}section{break-inside:avoid;break-after:page;border:none}}
"""
def main():
    pages=briefs();parts=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AttoFlow · Seven-role judgment classifiers</title><style>'+CSS+'</style><main><header><strong>AttoFlow</strong><span>Model evaluation</span></header><h1>Seven-role judgment classifiers</h1>']
    parts.append('<nav>'+''.join(f'<a href="#p{i}">{escape(p["title"])}</a>' for i,p in enumerate(pages,1))+'</nav>')
    for i,p in enumerate(pages,1):
        parts.append(f'<section id="p{i}"><h2>{escape(p["title"])}</h2><p class="sub">{escape(p["subtitle"])}</p>')
        for op,*args in p["items"]:
            if op=='text':parts.append('<p>'+escape(args[0])+'</p>')
            elif op=='heading':parts.append('<h3>'+escape(args[0])+'</h3>')
            elif op=='refs':parts.append('<div class="sources">'+''.join('<a href="'+escape(url,quote=True)+'">'+escape(k)+'</a>' for k,url in args[0])+'</div>')
            elif op=='table':
                cols,rows,_=args
                parts.append('<div class="scroll"><table><thead><tr>'+''.join('<th scope="col">'+escape(str(v))+'</th>' for v in cols)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in r)+'</tr>' for r in rows)+'</tbody></table></div>')
        parts.append('</section>')
    parts.append('</main></html>');tmp=OUT.with_suffix('.part');tmp.write_text('\n'.join(parts));tmp.replace(OUT)
    print('Rendered',len(pages),'sections')
if __name__=='__main__':main()
