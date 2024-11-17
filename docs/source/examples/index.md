# Code examples

NIR can be used to *export* or *import* models.
*Exporting* is when you convert a model from a source platform to NIR, and *importing* is when you convert a model from NIR to a target platform.
One typical workflow is to *export* a model from a simulator and *import* it to a hardware platform.

In the menu, you see examples for how to use NIR with your favorite framework.
But note that some frameworks only support importing or exporting.

## Writing to and reading from files with NIR
While NIR is typically integrated into your favorite framework, NIR supports writing to and reading from files directly.
This is useful when you want to send a model over email, store it for later, or share it with a colleague.

### Writing to a file
To write a model to a file, use the `nir.write` function.
Note that this requires you to provide a NIR model, so you need to find a way to convert your model to NIR within your framework.
The `nir.write` function takes two arguments: the file path and the model to write.
```python
import nir
my_nir_graph = ...
nir.write("my_graph.nir", my_model)
```

### Reading from a file
To read a model from a file, use the `nir.read` function.
This function takes a single argument: the file path.
```python
import nir
imported_graph = nir.read("my_graph.nir")
```

This gives you a NIR model, which then needs to be converted to your framework's model format.
The NIR graph itself is just a data structure.