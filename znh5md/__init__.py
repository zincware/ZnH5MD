"""ZnH5MD: A Zincwarecode package.

License
-------
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincwarecode Project.

Contact Information
-------------------
email: zincwarecode@gmail.com
github: https://github.com/zincware
web: https://zincwarecode.com/
"""
from znh5md.templates.espresso import EspressoH5MD
from znh5md.templates.lammps import LammpsH5MD

__all__ = [LammpsH5MD.__name__, EspressoH5MD.__name__]

__version__ = "0.1.0"
