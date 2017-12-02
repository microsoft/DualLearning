@rem =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
@rem Windows batch file to use Theano on GCR
@rem
@rem Updated: April 7, 2016
@rem =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

@rem set the PATH system variable
@rem Start from the 26th letter
set working_sub_dir=%cd:~26%

set PATH=^
C:\Windows\system32;^
C:\Windows\System32\Wbem;^
C:\Windows\System32\WindowsPowerShell\v1.0\;^
C:\Windows;^
C:\Program Files\Microsoft HPC Pack 2012\Bin\;^
C:\Program Files\Microsoft MPI\Bin\;^
C:\Program Files (x86)\Windows Kits\8.1\Windows Performance Toolkit\

pushd \\gcr\Scratch\RR1\v-yixia\Theano
set ToolkitFolderDriver=%cd%

@rem set the environment variable for the CUDA 7.5 Toolkit
rem set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5 rem the old version
set CUDA_HOME=%ToolkitFolderDriver%\CUDA\v7.0+cudnn4008
set CUDA_BIN=%CUDA_HOME%\bin
set CUDA_INCLUDE=%CUDA_HOME%\include
set CUDA_LIB=%CUDA_HOME%\lib\x64
set CUDA_LIBNVVP=%CUDA_HOME%\libnvvp

@rem add all CUDA Toolkit folders to the PATH system variable
set PATH=^
%CUDA_HOME%;^
%CUDA_BIN%;^
%CUDA_INCLUDE%;^
%CUDA_LIB%;^
%CUDA_LIBNVVP%;^
%PATH%

@echo %PATH%

@rem setting up VC complier

@rem =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
@rem connect to the shared toolkit folder \\gcr\Tools\Shared_Toolkits\Theano
rem pushd \\gcr\Tools\Shared_Toolkits\Theano



@rem unset these variables
@set Framework40Version=
@set FrameworkDIR32=
@set FrameworkVersion32=
@set FSHARPINSTALLDIR=
@set VSINSTALLDIR=
@set WindowsSDK_ExecutablePath_x64=
@set WindowsSDK_ExecutablePath_x86=

@set VCINSTALLDIR=%ToolkitFolderDriver%\VS_portable\Microsoft Visual Studio 12.0v2\VC\
@set WindowsSdkDir=%ToolkitFolderDriver%\Windows Kits\8.1v2\

:amd64

@rem set Windows SDK include/lib path
@rem --------------------------------------------------
if not "%WindowsSdkDir%" == "" @set PATH=%WindowsSdkDir%bin\x64;%WindowsSdkDir%bin\x86;%PATH%
if not "%WindowsSdkDir%" == "" @set INCLUDE=%WindowsSdkDir%include\shared;%WindowsSdkDir%include\um;%WindowsSdkDir%include\winrt;%INCLUDE%
if not "%WindowsSdkDir%" == "" @set LIB=%WindowsSdkDir%lib\winv6.3\um\x64;%LIB%
if not "%WindowsSdkDir%" == "" @set LIBPATH=%WindowsLibPath%;%ExtensionSDKDir%\Microsoft.VCLibs\14.0\References\CommonConfiguration\neutral;%LIBPATH%

@rem set the environment variables for Microsoft Visual Studio
@rem --------------------------------------------------
@rem PATH
@rem --------------------------------------------------
if exist "%VCINSTALLDIR%VCPackages" set PATH=%VCINSTALLDIR%VCPackages;%PATH%
if exist "%VCINSTALLDIR%BIN\amd64" set PATH=%VCINSTALLDIR%BIN\amd64;%PATH%
@rem --------------------------------------------------
@rem INCLUDE
@rem --------------------------------------------------
if exist "%VCINSTALLDIR%ATLMFC\INCLUDE" set INCLUDE=%VCINSTALLDIR%ATLMFC\INCLUDE;%INCLUDE%
if exist "%VCINSTALLDIR%INCLUDE" set INCLUDE=%VCINSTALLDIR%INCLUDE;%INCLUDE%
@rem --------------------------------------------------
@rem LIB
@rem --------------------------------------------------
if exist "%VCINSTALLDIR%ATLMFC\LIB\amd64" set LIB=%VCINSTALLDIR%ATLMFC\LIB\amd64;%LIB%
if exist "%VCINSTALLDIR%LIB\amd64" set LIB=%VCINSTALLDIR%LIB\amd64;%LIB%
@rem --------------------------------------------------
@rem LIBPATH
@rem --------------------------------------------------
if exist "%VCINSTALLDIR%ATLMFC\LIB\amd64" set LIBPATH=%VCINSTALLDIR%ATLMFC\LIB\amd64;%LIBPATH%
if exist "%VCINSTALLDIR%LIB\amd64" set LIBPATH=%VCINSTALLDIR%LIB\amd64;%LIBPATH%

@rem set the environment variables for the cuDNN v4 (Feb 10, 2016) for CUDA 7.0 and later.
rem set CUDNN_PATH=%ToolkitFolderDriver%\Shared_Toolkits\Theano\CUDA\cudnn-4.0.7\cuda
rem set INCLUDE=%CUDNN_PATH%\include;%INCLUDE%
rem set LIB=%CUDNN_PATH%\lib\x64;%LIB%
rem set PATH=%CUDNN_PATH%\bin;%PATH%

set Platform=X64
set CommandPromptType=Native

@rem =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
@rem connect to your scratch storage \\gcr\Scratch\<location>\<alias>
@rem Note: Please copy ANACONDA2 to \\gcr\Scratch\<location>\<alias>\Anaconda2
@rem =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
pushd \\gcr\scratch\RR1\v-yixia
set CONDANETDRIVE=%cd:~0,2%

@rem set the environment variable for the Anaconda2
set ANACONDA2=%CONDANETDRIVE%\RR1\taoqin\Anaconda-gpu002
set ANACONDA2_SCRIPTS=%ANACONDA2%\Scripts
set ANACONDA2_BIN=%ANACONDA2%\Library\bin

@rem add Anaconda2 folders to the PATH system variable
set PATH=^
%ANACONDA2%;^
%ANACONDA2_BIN%;^
%ANACONDA2_SCRIPTS%;^
%PATH%

@echo %PATH%

@rem example files from DeepLearningTutorials are available at \\gcr\Tools\Shared_Toolkits\Theano\Examples
set PROJDRIVE=%CONDANETDRIVE%
set MYHOME=%PROJDRIVE%\RR1\v-yixia
set PROJHOME=%MYHOME%\%working_sub_dir%

%PROJDRIVE%

cd %PROJHOME%

@rem setup theano env (generate .theanorc.txt)
call python gen_theanorc.py %ANACONDA2% .theanorc.txt
del %userprofile%\.theanorc.txt /Q /F
copy .theanorc.txt %userprofile% /Y

call python write_script.py %*

call worker.bat

@echo delete theano env
del %userprofile%\.theanorc.txt /Q /F

popd

popd


