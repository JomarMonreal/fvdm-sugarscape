import json
import cProfile
import pstats
import io
import sugarscape

# Minimal set of defaults needed to init Sugarscape without the full run_experiments setup
conf = json.load(open('configs/test_homo_fvdm_selfish.json'))
conf['seed'] = 42
conf['timesteps'] = 200
conf['headlessMode'] = True
conf['logfile'] = None
conf['agentStartingSugar'] = [10, 40]
conf['agentStartingSpice'] = [10, 40]
conf['keepAlivePostExtinction'] = False
conf['startingDiseases'] = 50
conf['agentTagging'] = True
conf['environmentSeasonInterval'] = 0
conf['environmentPollutionDiffusionDelay'] = 0
conf['environmentSugarConsumptionPollutionFactor'] = 0
conf['environmentSpiceConsumptionPollutionFactor'] = 0

pr = cProfile.Profile()
pr.enable()

s = sugarscape.Sugarscape(conf)
s.updateRuntimeStats()
for t in range(conf['timesteps']):
    s.doTimestep()
    if len(s.agents) == 0 or s.end:
        break

pr.disable()

sio = io.StringIO()
ps = pstats.Stats(pr, stream=sio).sort_stats('cumulative')
ps.print_stats(30)
print(sio.getvalue())
