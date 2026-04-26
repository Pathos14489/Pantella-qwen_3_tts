# Qwen3 TTS

This is an addon for Pantella that adds Qwen3 TTS as a tts_type that can be use for Fallout TTW and Skyrim SE/AE.

# How To Install

1. Find the latest release and follow the installation instructions [over here](https://github.com/Pathos14489/Pantella-qwen_3_tts/releases) to download the addon and place it in your Pantella addons folder.
2. Run `pip install -r requirements.txt` in the qwen_3 folder to install the required dependencies for the addon. You can do this in a terminal or command prompt, just make sure to navigate to the qwen_3 folder first before running the command. If you're using the launcher, make sure to install the requirements in the Python environment that the launcher is using by running `./python-3.10.11-embed/python.exe -m pip install -r ./repositories/Pathos14489_Pantella_dev/addons/qwen_3/requirements.txt` from the launcher directory.
3. Enjoy! Enable it in your tts_engines section in your config.json to enable support by adding "qwen_3_tts" to the list of enabled TTS engines. 

# Credits

Uses the code from [this repo](https://github.com/Suaroman/qwen3-tts). Big thanks to the Qwen3 Devs for open sourcing the code with such a permissive license, made making this addon very simple!

# Incompatibilities

Incompatible with the [Pantella-indextts2](https://github.com/Pathos14489/Pantella-indextts2) addon due to both addons using different versions of the transformers library which causes dependency conflicts.

# Need Help?

If you need support please reach out in the [Pantella Discord](https://discord.gg/M7Zw8mBY6r).