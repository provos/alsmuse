# Tools for helping with A/V scripts for music videos

I want to know in precise time where meaningful transitions happen in an audio track. How they relate to the timing of the lyrics and how that should inform what should happen in a music video.

I use Ableton Live and want to analyze Ableton live set lists (ALS) files. They are gzipped XML.

My tracks usually have a MIDI track that is called structure. it will have clips labeled INTRO, VERSE, CHORUS, BRIDGE, OUTRO, etc. We can extract the exact timing and durations based on the bpm and time markers in the ALS.

There may be MIDI tracks for drums where we can see where the kick, snare and other percussion instruments hit.
There may be audio tracks for kick and snare where we can analyze transients.

There will be other tracks using MIDI for synthesizers. Some of those may be leads and others may be pads or special effects.

Usually, I have lyrics file with designated verse, chorus and bridge sections. Sometimes the lyrics are timestamped but not initially.

The output should be the A column of an A/V table. That contains lyrics(?), transitions and hits(?)