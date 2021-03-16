import setuptools

setuptools.setup(
	name = 'wheeledSim',
	packages=['wheeledSim','wheeledRobots.clifford'],
	include_package_data=True,
	package_data={"":['*.sdf','meshes/*']}
)