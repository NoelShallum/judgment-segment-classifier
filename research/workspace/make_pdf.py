"""Lay out a four-page, fully offline team brief from verified score records."""
from datetime import datetime
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph, Table, TableStyle

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parents[1] / "outputs/judgment-role/AttoFlow-judgment-role-results.pdf"
FONT = Path("/usr/share/fonts/truetype/ibm-plex")
for face, name in [("JRText","IBMPlexSans-Regular"),("JRBold","IBMPlexSans-SemiBold"),
                   ("JRSerif","IBMPlexSerif-Regular"),("JRMono","IBMPlexMono-Regular")]:
    pdfmetrics.registerFont(TTFont(face,str(FONT/(name+".ttf"))))
pdfmetrics.registerFontFamily("JRText",normal="JRText",bold="JRBold",italic="JRText",boldItalic="JRBold")
INK = "#1A2C32"
TEAL = "#066456"
MUTED = "#475D61"
LINE = "#D5DEDA"
LIGHT = "#ECF3EF"
W,H=A4
LEFT=43
WIDTH=W-2*LEFT


def record(name):
    path=ROOT/"results"/name
    return json.loads(path.read_text()) if path.is_file() else None


def style(face,size,leading,color=INK):
    return ParagraphStyle(face+str(size),fontName=face,fontSize=size,leading=leading,
                          textColor=colors.HexColor(color),allowOrphans=1,allowWidows=1)


def link(label, href):
    return f'<link href="{escape(href)}" color="{TEAL}"><u>{escape(label)}</u></link>'


class Page:
    def __init__(self, pdf, number, title, subtitle):
        self.c=pdf;self.n=number;self.y=H-78
        pdf.setFillColor(colors.HexColor(TEAL));pdf.setFont("JRBold",12)
        pdf.drawString(LEFT,H-33,"AttoFlow")
        pdf.setFillColor(colors.HexColor(MUTED));pdf.setFont("JRText",9)
        pdf.drawRightString(W-LEFT,H-32,"Model evaluation   /   "+datetime.now().strftime("%d %B %Y"))
        pdf.setStrokeColor(colors.HexColor(LINE));pdf.line(LEFT,H-45,W-LEFT,H-45)
        self.text(title,style("JRSerif",27,30),7)
        self.text(subtitle,style("JRText",11.3,15.6,MUTED),14)
        pdf.bookmarkPage(f"p{number}");pdf.addOutlineEntry(title,f"p{number}")
    def text(self, text, fmt=None, gap=7):
        para=Paragraph(text,fmt or style("JRText",11.1,16.0))
        _,height=para.wrap(WIDTH,1000)
        self.y-=height;para.drawOn(self.c,LEFT,self.y);self.y-=gap
    def section(self,title):
        self.y-=3;self.text(title,style("JRBold",14,19,TEAL),5)
    def table(self, labels, rows, widths):
        styles=[style("JRBold",9.5,12.1,TEAL),style("JRText",10.2,13.7)]
        data=[[Paragraph(str(v),styles[0]) for v in labels]]
        for row in rows:
            data.append([Paragraph(str(v),styles[1]) for v in row])
        table=Table(data,colWidths=[WIDTH*x for x in widths],hAlign="LEFT")
        table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor(LIGHT)),
                ("VALIGN",(0,0),(-1,-1),"TOP"),("TOPPADDING",(0,0),(-1,-1),5),
                ("BOTTOMPADDING",(0,0),(-1,-1),5),("LEFTPADDING",(0,0),(-1,-1),6),
                ("RIGHTPADDING",(0,0),(-1,-1),6),("LINEBELOW",(0,0),(-1,-1),.32,colors.HexColor(LINE))]))
        _,height=table.wrap(WIDTH,1000);self.y-=height;table.drawOn(self.c,LEFT,self.y);self.y-=10
    def refs(self,*pairs):
        self.text("  ·  ".join(link(x,y) for x,y in pairs),style("JRText",8.6,12,MUTED),6)
    def finish(self):
        if self.y<45:raise ValueError(f"Page {self.n} overflows, bottom {self.y:.1f}")
        self.c.setStrokeColor(colors.HexColor(LINE));self.c.line(LEFT,35,W-LEFT,35)
        self.c.setFont("JRText",8.5);self.c.setFillColor(colors.HexColor(MUTED))
        self.c.drawString(LEFT,22,"AttoFlow   /   Judgment rhetorical roles")
        self.c.drawRightString(W-LEFT,22,f"{self.n}  /  {self.total}")
        self.c.showPage()
        print("Page",self.n,"bottom",round(self.y))


def main():
    from comparison_brief import briefs
    OUT.parent.mkdir(exist_ok=True,parents=True)
    tmp=OUT.with_suffix(".part")
    pdf=Canvas(str(tmp),pagesize=A4,pageCompression=1)
    pdf.setTitle("AttoFlow | Judgment-role classifier research stocktake");pdf.setAuthor("AttoFlow research briefing")
    pages=briefs()
    for i,description in enumerate(pages,1):
        p=Page(pdf,i,description["title"],description["subtitle"]);p.total=len(pages)
        for op,*args in description["items"]:
            if op=="text":p.text(args[0])
            elif op=="heading":p.section(args[0])
            elif op=="table":p.table(*args)
            elif op=="refs":p.refs(*args[0])
        p.finish()
    pdf.save();tmp.replace(OUT)
if __name__=="__main__":main()
