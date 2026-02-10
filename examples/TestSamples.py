import soundfile as sf
from workspace.QwenTTSModel import QwenTTSManager

""" Single Ellipsis (...): Creates a short, reflective pause.
Double Ellipsis (......): Forces a significantly longer, dramatic break—perfect for John Wayne’s mid-sentence stops.
Dashes (--): Use these to simulate a sharp, "punctuated" break before a final word or phrase.
https://www.youtube.com/watch?v=pXMCohnRbqQ&t=11s """
myInputText = "I think......you'd better...... WATER my horse."
#myInputText = "我想……你最好……去给我的马喂点水。" 
#myInputText = "Four score and seven years ago our fathers brought forth on this continent a new nation, conceived in liberty, and dedicated to the proposition that all men are created equal. “Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great battlefield of that war. "
#myInputText = "八十七年前，我们的先辈们在这片大陆上创建了一个新国家，它孕育于自由之中，并致力于这样一个原则：人人生而平等。 如今，我们正卷入一场伟大的内战，这场战争考验着这个国家，也考验着任何一个以同样理念建立并致力于同样原则的国家，是否能够长久存续。我们聚集在这场战争的一片伟大战场上。我们来到这里，是为了将这片战场的一部分奉献出来，作为那些为国家生存而献出生命的人们的最终安息之地。我们这样做是完全恰当和应该的"
# Use manual pause punctuation directly in the text
#myInputText = "LISTEN...... pilgrim. I said...... don't *ever*...... touch that saddle again. That is - MY - horse, pilgrim. Now, I *reckon*...... that's enough talk. Weeeell...... pilgram...... I reckon it's time--...... you moved on."
#myInputText = "听着......朝圣者。我说过......不要*再*碰那马鞍了。那是——我的——马，朝圣者。现在，我*想*......话就到此为止。好吧......朝圣者......我想是时候——......你该离开了。"
""" myInputText = ("Now, I want you to listen here, pilgrim. "
"I said, don't ever touch that saddle again. "
"That is MY horse, pilgrim. "
"Now, I reckon that's enough talk. "
"Weeeell, pilgrim, I reckon it's time you moved on.") """
#myInputText = "^shhh^...... be quiet, they're right outside....... we don't want to startle them. Just stay calm and follow my lead...... slowly now...... we'll get through this...... together. GET OUT OF HERE! NOW!"
myInputText = "Well, pilgrim, I reckon it's time you moved on."

#myOutputFile = "myWavOutputs/clone/iThink.wav"
myOutputFile = "/data/audio/output/qwen/clone/pilgrim2.wav"

#myOutputFile = "myWavOutputs/clone/Gettysburg.wav"
myLanguage = "English"
#myLanguage = "Chinese"
#myInstruct = "用低沉、沙哑、缓慢且威严的声音说话。带有颗粒感和节奏感。"
#myInstruct ="Speak with a deep, gravelly, authoritative grit, but perfectly in Chinese."
#myInstruct="A deep, gravelly, and authoritative Western-style voice speaking in Mandarin Chinese."
myInstruct = "A deep, gravelly, rhythmic Western drawl. Slow, punctuated delivery with a weary, authoritative grit."
#myInstruct= "Speak with a loud, booming, and commanding shout. High energy and authoritative."

myRefAudio = "/data/audio/input/JohnWayne/YouTalkThinkTooMuch.wav" 
myRefText = "Yyou talk too much...THINK too much"

# Refined instruct tag for timing
""" myInstruct = (
    "A deep, gravelly voice with an authoritative grit. "
    "Use a slow, rhythmic delivery with very deliberate, long pauses for dramatic effect."
) """

#myTemperature = 0.9  # Lowering temperature slightly helps stabilize the specific rhythm

# 1. Instantiate the Base model for cloning
ttsm = QwenTTSManager()
# 2. Define reference material
# 3. Create the reusable VoiceClonePromptItem
ttsm.set_voice_clone(ref_audio=myRefAudio, ref_text=myRefText)
# 4. Generate speech with an instruction/style tag
wavs, sr = ttsm.generate(myInputText, myInstruct)
sf.write(myOutputFile, wavs[0], sr)
