.. rendering:


Rendering
==========


Bundles currently support two renders: text and plot mode; but you can always define you own render mode. When you ask a bundle to render, it will call each agent's as well as the task's render method. 

You have to specify which mode you want when calling the render method. For example

.. code-block:: python

   bundle.render('text-plot', *args, **kwargs)

will trigger text and plot modes.

* The point of text mode is to print to stdout some useful diagnostics.
* In plot mode, a matplotlib figure is started by the bundle and the axes are passed to agents and task.

 The expected signature for text mode for agents and task is 

 .. code-block:: python

   def render(self, mode =None, ax_user = None, ax_assistant = None, ax_task = None, **kwargs)
