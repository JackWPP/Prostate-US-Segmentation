
PS C:\Users\WPP_JKW\Prostate-US-Segmentation> python src\visualize.py
Starting Gradio interface...
Open the following URL in your browser to view the UI.

* Running on local URL:  http://127.0.0.1:7860
* Running on public URL: https://45e826f4e822ba5d15.gradio.live

This share link expires in 72 hours. For free permanent hosting and GPU upgrades, run `gradio deploy` from the terminal in the working directory to deploy to Hugging Face Spaces (https://huggingface.co/spaces)
ERROR:    Exception in ASGI application
Traceback (most recent call last):
  File "C:\Users\WPP_JKW\AppData\Roaming\Python\Python312\site-packages\uvicorn\protocols\http\httptools_impl.py", line 409, in run_asgi
    result = await app(  # type: ignore[func-returns-value]
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\WPP_JKW\AppData\Roaming\Python\Python312\site-packages\uvicorn\middleware\proxy_headers.py", line 60, in __call__
    return await self.app(scope, receive, send)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\WPP_JKW\AppData\Roaming\Python\Python312\site-packages\fastapi\applications.py", line 1054, in __call__
    await super().__call__(scope, receive, send)
  File "C:\Users\WPP_JKW\AppData\Roaming\Python\Python312\site-packages\starlette\applications.py", line 112, in __call__
    await self.middleware_stack(scope, receive, send)
  File "C:\Users\WPP_JKW\AppData\Roaming\Python\Python312\site-packages\starlette\middleware\errors.py", line 187, in __call__
    raise exc
  File "C:\Users\WPP_JKW\AppData\Roaming\Python\Python312\site-packages\starlette\middleware\errors.py", line 165, in __call__
    await self.app(scope, receive, _send)
  File "C:\Users\WPP_JKW\AppData\Roaming\Python\Python312\site-packages\gradio\route_utils.py", line 789, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\WPP_JKW\AppData\Roaming\Python\Python312\site-packages\starlette\middleware\exceptions.py", line 62, in __call__
    await wrap_app_handling_exceptions(self.app, conn)(scope, receive, send)
  File "C:\Users\WPP_JKW\AppData\Roaming\Python\Python312\site-packages\starlette\_exception_handler.py", line 53, in wrapped_app
    raise exc
  File "C:\Users\WPP_JKW\AppData\Roaming\Python\Python312\site-packages\starlette\_exception_handler.py", line 42, in wrapped_app
    await app(scope, receive, sender)
  File "C:\Users\WPP_JKW\AppData\Roaming\Python\Python312\site-packages\starlette\routing.py", line 714, in __call__
    await self.middleware_stack(scope, receive, send)
  File "C:\Users\WPP_JKW\AppData\Roaming\Python\Python312\site-packages\starlette\routing.py", line 734, in app
    await route.handle(scope, receive, send)
  File "C:\Users\WPP_JKW\AppData\Roaming\Python\Python312\site-packages\starlette\routing.py", line 288, in handle
    await self.app(scope, receive, send)
  File "C:\Users\WPP_JKW\AppData\Roaming\Python\Python312\site-packages\starlette\routing.py", line 76, in app
    await wrap_app_handling_exceptions(app, request)(scope, receive, send)
  File "C:\Users\WPP_JKW\AppData\Roaming\Python\Python312\site-packages\starlette\_exception_handler.py", line 53, in wrapped_app
    raise exc
  File "C:\Users\WPP_JKW\AppData\Roaming\Python\Python312\site-packages\starlette\_exception_handler.py", line 42, in wrapped_app
    await app(scope, receive, sender)
  File "C:\Users\WPP_JKW\AppData\Roaming\Python\Python312\site-packages\starlette\routing.py", line 73, in app
    response = await f(request)
               ^^^^^^^^^^^^^^^^
  File "C:\Users\WPP_JKW\AppData\Roaming\Python\Python312\site-packages\fastapi\routing.py", line 301, in app
    raw_response = await run_endpoint_function(
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\WPP_JKW\AppData\Roaming\Python\Python312\site-packages\fastapi\routing.py", line 214, in run_endpoint_function
    return await run_in_threadpool(dependant.call, **values)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\WPP_JKW\AppData\Roaming\Python\Python312\site-packages\starlette\concurrency.py", line 37, in run_in_threadpool
    return await anyio.to_thread.run_sync(func)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\WPP_JKW\AppData\Roaming\Python\Python312\site-packages\anyio\to_thread.py", line 33, in run_sync
    return await get_asynclib().run_sync_in_worker_thread(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\WPP_JKW\AppData\Roaming\Python\Python312\site-packages\anyio\_backends\_asyncio.py", line 877, in run_sync_in_worker_thread
    return await future
           ^^^^^^^^^^^^
  File "C:\Users\WPP_JKW\AppData\Roaming\Python\Python312\site-packages\anyio\_backends\_asyncio.py", line 807, in run
    result = context.run(func, *args)
             ^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\WPP_JKW\AppData\Roaming\Python\Python312\site-packages\gradio\routes.py", line 584, in main
    gradio_api_info = api_info(request)
                      ^^^^^^^^^^^^^^^^^
  File "C:\Users\WPP_JKW\AppData\Roaming\Python\Python312\site-packages\gradio\routes.py", line 615, in api_info
    api_info = utils.safe_deepcopy(app.get_blocks().get_api_info())
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\WPP_JKW\AppData\Roaming\Python\Python312\site-packages\gradio\blocks.py", line 3048, in get_api_info
    python_type = client_utils.json_schema_to_python_type(info)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\WPP_JKW\AppData\Roaming\Python\Python312\site-packages\gradio_client\utils.py", line 931, in json_schema_to_python_type
    type_ = _json_schema_to_python_type(schema, schema.get("$defs"))
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\WPP_JKW\AppData\Roaming\Python\Python312\site-packages\gradio_client\utils.py", line 986, in _json_schema_to_python_type
    f"{n}: {_json_schema_to_python_type(v, defs)}{get_desc(v)}"
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\WPP_JKW\AppData\Roaming\Python\Python312\site-packages\gradio_client\utils.py", line 993, in _json_schema_to_python_type
    f"str, {_json_schema_to_python_type(schema['additionalProperties'], defs)}"
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\WPP_JKW\AppData\Roaming\Python\Python312\site-packages\gradio_client\utils.py", line 939, in _json_schema_to_python_type
    type_ = get_type(schema)
            ^^^^^^^^^^^^^^^^
  File "C:\Users\WPP_JKW\AppData\Roaming\Python\Python312\site-packages\gradio_client\utils.py", line 898, in get_type
    if "const" in schema:
       ^^^^^^^^^^^^^^^^^
TypeError: argument of type 'bool' is not iterable
