@echo off
rem LLMBench GUI を起動する（コンソールウィンドウを出さない）
cd /d "%~dp0"
start "" pythonw -m llmbench gui
