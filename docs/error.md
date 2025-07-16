PS C:\Users\WPP_JKW\Prostate-US-Segmentation> pip install -r requirements.txt
Defaulting to user installation because normal site-packages is not writeable
Looking in indexes: https://mirrors.aliyun.com/pypi/simple/
Requirement already satisfied: nibabel in c:\users\wpp_jkw\appdata\roaming\python\python312\site-packages (from -r requirements.txt (line 1)) (5.2.1)
Requirement already satisfied: numpy in c:\users\wpp_jkw\appdata\roaming\python\python312\site-packages (from -r requirements.txt (line 2)) (2.2.6)
Requirement already satisfied: opencv-python in c:\users\wpp_jkw\appdata\roaming\python\python312\site-packages (from -r requirements.txt (line 3)) (4.12.0.88)
Requirement already satisfied: albumentations in c:\users\wpp_jkw\appdata\roaming\python\python312\site-packages (from -r requirements.txt (line 4)) (2.0.8)
Requirement already satisfied: tqdm in c:\users\wpp_jkw\appdata\roaming\python\python312\site-packages (from -r requirements.txt (line 5)) (4.67.1)
Requirement already satisfied: torch in c:\users\wpp_jkw\appdata\roaming\python\python312\site-packages (from -r requirements.txt (line 6)) (2.7.1+cu128)
Requirement already satisfied: torchvision in c:\users\wpp_jkw\appdata\roaming\python\python312\site-packages (from -r requirements.txt (line 7)) (0.22.1+cu128)
Requirement already satisfied: matplotlib in c:\users\wpp_jkw\appdata\roaming\python\python312\site-packages (from -r requirements.txt (line 8)) (3.10.3)
Requirement already satisfied: gradio in c:\users\wpp_jkw\appdata\roaming\python\python312\site-packages (from -r requirements.txt (line 9)) (5.16.0)
Collecting mamba-ssm (from -r requirements.txt (line 10))
  Downloading https://mirrors.aliyun.com/pypi/packages/1d/c7/6e21ecece28e6d625f42e708c7523cd78ec82d1622f98562f82bf02748b7/mamba_ssm-2.2.4.tar.gz (91 kB)
  Installing build dependencies ... done
  Getting requirements to build wheel ... error
  error: subprocess-exited-with-error

  × Getting requirements to build wheel did not run successfully.
  │ exit code: 1
  ╰─> [26 lines of output]
      C:\Users\WPP_JKW\AppData\Local\Temp\pip-build-env-m57afdx7\overlay\Lib\site-packages\torch\_subclasses\functional_tensor.py:276: UserWarning: Failed to initialize NumPy: No module named 'numpy' (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\pytorch\torch\csrc\utils\tensor_numpy.cpp:81.)
        cpu = _conversion_method_template(device=torch.device("cpu"))
      `<string>`:118: UserWarning: mamba_ssm was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, only images whose names contain 'devel' will provide nvcc.

    torch.__version__  = 2.7.1+cpu

    Traceback (most recent call last):
        File "C:\Users\WPP_JKW\AppData\Roaming\Python\Python312\site-packages\pip\_vendor\pyproject_hooks\_in_process\_in_process.py", line 389, in `<module>`
          main()
        File "C:\Users\WPP_JKW\AppData\Roaming\Python\Python312\site-packages\pip\_vendor\pyproject_hooks\_in_process\_in_process.py", line 373, in main
          json_out["return_val"] = hook(**hook_input["kwargs"])
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File "C:\Users\WPP_JKW\AppData\Roaming\Python\Python312\site-packages\pip\_vendor\pyproject_hooks\_in_process\_in_process.py", line 143, in get_requires_for_build_wheel
          return hook(config_settings)
                 ^^^^^^^^^^^^^^^^^^^^^
        File "C:\Users\WPP_JKW\AppData\Local\Temp\pip-build-env-m57afdx7\overlay\Lib\site-packages\setuptools\build_meta.py", line 331, in get_requires_for_build_wheel
          return self._get_build_requires(config_settings, requirements=[])
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File "C:\Users\WPP_JKW\AppData\Local\Temp\pip-build-env-m57afdx7\overlay\Lib\site-packages\setuptools\build_meta.py", line 301, in _get_build_requires
          self.run_setup()
        File "C:\Users\WPP_JKW\AppData\Local\Temp\pip-build-env-m57afdx7\overlay\Lib\site-packages\setuptools\build_meta.py", line 317, in run_setup
          exec(code, locals())
        File "`<string>`", line 188, in `<module>`
      NameError: name 'bare_metal_version' is not defined
      [end of output]

  note: This error originates from a subprocess, and is likely not a problem with pip.
error: subprocess-exited-with-error

× Getting requirements to build wheel did not run successfully.
│ exit code: 1
╰─> See above for output.

note: This error originates from a subprocess, and is likely not a problem with pip.
