# Python 代码风格指南

本项目遵循 Python PEP 8 代码风格规范。

## 编码规范

### 1. 命名规范

- **类名**: 使用大驼峰命名法 (CamelCase)
  ```python
  class DataProcessor:
      pass
  ```

- **函数和变量**: 使用小写 + 下划线 (snake_case)
  ```python
  def process_data():
      pass

  user_name = "John"
  ```

- **常量**: 使用全大写 + 下划线
  ```python
  MAX_SIZE = 100
  COLOR_MAP = {"key": "value"}
  ```

- **私有方法/变量**: 使用单下划线前缀
  ```python
  def _internal_helper():
      pass
  ```

### 2. 代码格式

- **缩进**: 使用 4 个空格（不使用 Tab）
- **行宽**: 最大 100 字符
- **空行**:
  - 函数之间空 2 行
  - 类中方法之间空 1 行
  - 逻辑段落之间空 1 行

### 3. 导入规范

- 标准库导入放在最前面
- 第三方库导入紧随其后
- 本地模块导入放在最后
- 每组导入之间空一行

```python
import os
import sys

import pandas as pd
import numpy as np

from .local_module import MyClass
```

### 4. 文档字符串

所有公共函数和类都应包含文档字符串：

```python
def calculate_average(numbers):
    """计算数字列表的平均值。

    Args:
        numbers: 数字列表

    Returns:
        平均值（float）
    """
    return sum(numbers) / len(numbers)
```

### 5. 类型注解

鼓励使用类型注解：

```python
def greet(name: str, age: int) -> str:
    return f"Hello, {name}. You are {age} years old."
```

## 自动化工具

### 使用 Black 格式化代码

```bash
pip install black
black .
```

### 使用 Flake8 检查代码

```bash
pip install flake8
flake8 .
```

### 使用 isort 排序导入

```bash
pip install isort
isort .
```

## VS Code 配置

在 `.vscode/settings.json` 中添加：

```json
{
    "python.formatting.provider": "black",
    "python.linting.flake8Enabled": true,
    "editor.formatOnSave": true,
    "editor.rulers": [100]
}
```
