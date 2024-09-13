from moviepy.editor import VideoFileClip

clip = VideoFileClip('demo_video.mp4').subclip(8, 25)

clip.write_gif('demo.gif', fps=60, program='ffmpeg')
