pandoc -H disable_float.tex \
  -f markdown \
  -t pdf \
  -V colorlinks=true \
  -V linkcolor=blue \
  -V urlcolorblue \
  -V toccolor=gray \
  --highlight-style tango \
  $1 \
  -H break_lines.tex \
  -o $2