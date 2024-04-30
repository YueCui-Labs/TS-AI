import vtk

# Create a reader for the VTK file
reader = vtk.vtkDataSetReader()
reader.SetFileName('Sphere.642.vtk')
reader.Update()

# Create a renderer, render window, and interactor
renderer = vtk.vtkRenderer()
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)

# Create an actor and add it to the renderer
actor = vtk.vtkActor()
mapper = vtk.vtkDataSetMapper()
mapper.SetInputConnection(reader.GetOutputPort())
actor.SetMapper(mapper)
renderer.AddActor(actor)

# Set up the camera and start the interaction
renderer.ResetCamera()
interactor.Initialize()
render_window.Render()
interactor.Start()