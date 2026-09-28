import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ['MPLBACKEND'] = 'Agg'
import matplotlib.pyplot as plt
import app


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.previous_dir = Path.cwd()
        os.chdir(self.temp.name)
        self.files = []
        for index, unit in enumerate(('N', 'daN', 'kN'), 1):
            path = Path(f'{index}.csv')
            path.write_text(f'Time;Displacement;Force\ns;mm;{unit}\n0;0;0\n1;{index};{index}\n2;{index + 1};0\n')
            self.files.append(str(path))

    def tearDown(self):
        plt.close('all')
        os.chdir(self.previous_dir)
        self.temp.cleanup()

    def test_experimental_outputs_never_compute_model(self):
        with patch.object(app.MATER_306, 'compute_response', side_effect=AssertionError('Simulation called')), patch.object(plt, 'show'):
            figures = app.experimental_analysis(*self.files)
        self.assertEqual(plt.get_fignums(), [1, 2, 6])
        self.assertEqual([max(line.get_ydata()) for line in figures[0].axes[0].lines], [1, 20, 3000])
        self.assertEqual({p.name for p in Path('output').iterdir()},
                         {'courbes_essais.png', 'tableau_rivet_bleu.png', 'mohr_colore.png'})

    def test_simulation_outputs_and_preview(self):
        with patch.object(plt, 'show'), patch.object(app, 'plot_polar_failure_criteria', side_effect=AssertionError('Polar called')):
            app.main(*self.files)
            self.assertEqual(plt.get_fignums(), [3, 4, 5, 8])
            self.assertEqual({p.name for p in Path('output').iterdir()},
                             {'comparaison_matis.png', 'correlation_efforts.png', 'bilan_essais_simu.png', 'parametres_simulation.png'})
            fig = app.main(*self.files, save_outputs=False, preview_mode=True)
            self.assertEqual(len(fig.axes), 3)
            self.assertTrue(all(len(ax.lines) == 2 for ax in fig.axes))
            self.assertEqual(plt.get_fignums(), [])

    def test_repeated_embedded_experiments_and_invalid_input(self):
        for _ in range(2):
            figures = app.experimental_analysis(*self.files, save_outputs=False, embedded=True)
            self.assertEqual(len(figures), 3)
            self.assertEqual(plt.get_fignums(), [])
        with self.assertRaises(ValueError):
            app.experimental_analysis(self.files[0], 'missing.csv', self.files[2])
        self.assertEqual(plt.get_fignums(), [])


if __name__ == '__main__':
    unittest.main()
