# Prepare

## For Window User 
Installing aiortc on windows may sometimes requires Microsoft Build Tools for Visual C++ libraries installed. You can easily fix this error by installing any ONE of these choices:

While the error is calling for VC++ 14.0 - but newer versions of Visual C++ libraries works as well.

+ Microsoft [Build Tools for Visual Studio.](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16)

+ Alternative link to Microsoft [Build Tools for Visual Studio.](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019)

+ Offline installer: [vs_buildtools.exe](https://aka.ms/vs/16/release/vs_buildtools.exe)

Afterwards, Select: Workloads → Desktop development with C++, then for Individual Components, select only:

Windows 10 SDK
C++ x64/x86 build tools

For more Detail : [vidgear](https://abhitronix.github.io/vidgear/v0.2.3-stable/installation/source_install/)

## Install Dependency
` $ pip install -r requirements.txt `


# Run

1. Edit Detection Server address from [here](https://github.com/boostcampaitech2/final-project-level3-cv-04/blob/dfbc723180c52eb5c182c687983c6e01b377b317/web_server/app.py#L37)

2. Run Streamlit
` streamlit run app.py --server.port 6006 `
