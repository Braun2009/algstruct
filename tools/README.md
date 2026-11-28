# tools

`md_notation.py` 把文档里的 LaTeX 转成等宽记号：行内 `$...$` 变代码段，`$$...$$` 变 ```text 块。

原因：不支持的渲染器会把公式显示成源码。代码段与围栏块在任何渲染器下都正常，
而这种记号用等宽字体读起来一样清楚。

```text
python tools/md_notation.py            # 预览
python tools/md_notation.py --write    # 应用
```

转换是幂等的：重复运行不会再改动已转换的内容。
