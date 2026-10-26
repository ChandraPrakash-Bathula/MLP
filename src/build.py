import sys,os,re,base64
S=os.path.dirname(os.path.abspath(__file__))
csv=open(sys.argv[1]).read().strip()
IMG=os.path.join(S,'img')
def inline_imgs(s):
  # {{IMG:name}} -> data URI of img/name.png (blog figures)
  return re.sub(r'\{\{IMG:(\w+)\}\}', lambda m: 'data:image/png;base64,' + base64.b64encode(open(os.path.join(IMG, m.group(1) + '.png'),'rb').read()).decode(), s)
r=lambda f: inline_imgs(open(os.path.join(S,f)).read())
frag = (r('head.html') + r('body.html') +
  '\n<script>window.MathJax = { tex: { inlineMath: [["\\\\(", "\\\\)"]], displayMath: [["\\\\[", "\\\\]"]] }, svg: { fontCache: "global" }, options: { enableMenu: false } };</script>\n' +
  '<script async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-svg-full.min.js"></script>\n' +
  '<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/plotly.min.js"></script>\n' +
  '<script>\nconst HEART_CSV = `' + csv + '`;\n</script>\n' +
  '<script>\n' + r('core.js') + '\n</script>\n<script>\n' + r('app.js') + '\n</script>\n')
open(os.path.join(S,'mlp.html'),'w').write(frag)
head,rest = r('head.html'), r('body.html')
full = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n' + head +
  '</head>\n<body>\n' + frag[len(head):] + '</body>\n</html>\n')
open(os.path.join(S,'index.html'),'w').write(full)
print(len(frag), len(full))
