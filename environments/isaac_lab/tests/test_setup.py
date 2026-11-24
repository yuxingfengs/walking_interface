"""Offline checks for the Isaac runtime and NAO assets."""
from pathlib import Path
from argparse import Namespace
import sys
import tempfile
import unittest

SIM = Path(__file__).resolve().parents[1]
sys.path.extend([str(SIM), str(SIM/'setup')])
from asset_contract import LEG_JOINTS, validate_urdf


class IsaacSetupTests(unittest.TestCase):
    def test_standard_kit_modes_use_the_installed_lab_extensions(self):
        import os
        import tomllib
        from unittest.mock import patch
        from runtime_versions import project_experience

        with tempfile.TemporaryDirectory() as location, patch.dict(os.environ, {
                'HEADLESS': '0', 'LIVESTREAM': '0', 'ENABLE_CAMERAS': '0', 'XR': '0'}):
            lab = Path(location)
            apps = lab/'apps'
            apps.mkdir()
            addons = ('isaaclab_mimic', 'isaaclab_tasks', 'isaaclab_rl')
            for headless in (True, False):
                mode = 'headless' if headless else 'gui'
                vendor = apps/('isaaclab.python.headless.kit' if headless else 'isaaclab.python.kit')
                template = ('[dependencies]\n"omni.kit.telemetry" = {}\n'
                            f'"{mode}" = {{}}\n"isaaclab_assets" = {{order = 1000}}\n'
                            + ''.join(f'"{name}" = {{order = 1000}}\n' for name in addons))
                vendor.write_text(template)
                args = Namespace(headless=headless, experience='')
                for installed in ((), addons[:1], addons):
                    with self.subTest(mode=mode, installed=installed):
                        for name in installed:
                            manifest = lab/'source'/name/'config/extension.toml'
                            manifest.parent.mkdir(parents=True, exist_ok=True)
                            manifest.touch()
                        path = project_experience(args, {'isaac_lab_path': lab}, lab)
                        deps = tomllib.loads(Path(path).read_text())['dependencies']
                        self.assertEqual(set(deps), {mode, 'isaaclab_assets', *installed})
                        self.assertEqual(vendor.read_text(), template)
                for name in addons:
                    (lab/'source'/name/'config/extension.toml').unlink()
            self.assertEqual(project_experience(Namespace(experience='custom.kit'), {}, lab), 'custom.kit')

    def test_stl_alias_preserves_geometry_and_rejects_modified_alias(self):
        from prepare_asset import import_mesh_path
        with tempfile.TemporaryDirectory() as location:
            root = Path(location)
            mesh = root/'Torso_0.10.stl'
            original = b'solid torso\nendsolid torso\n'
            mesh.write_bytes(original)
            alias = import_mesh_path(mesh, root/'assets')
            self.assertRegex(alias.stem, r'^[A-Za-z_][A-Za-z0-9_]*$')
            self.assertEqual(alias.read_bytes(), original)
            self.assertEqual(mesh.read_bytes(), original)
            self.assertEqual(import_mesh_path(mesh, root/'assets'), alias)
            alias.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                import_mesh_path(mesh, root/'assets')
            valid = root/'Torso.stl'
            valid.write_bytes(original)
            self.assertEqual(import_mesh_path(valid, root/'assets'), valid)

    def test_warp_overlay_preserves_vendor_and_rejects_changed_native_files(self):
        from configure_warp_extension import build_extension
        from runtime_versions import WARP_VERSION, warp_extension_path, kit_arguments
        with tempfile.TemporaryDirectory() as location:
            root = Path(location)
            vendor, package = root/'vendor', root/'wheel'
            (vendor/'config').mkdir(parents=True)
            (vendor/'config/extension.toml').write_text('[package]\nversion = "1.7.1"\n')
            (vendor/'warp').mkdir()
            (vendor/'warp/old.dll').write_bytes(b'original')
            (package/'bin').mkdir(parents=True)
            (package/'__init__.py').write_text('version = "fixture"')
            for name in ('warp.dll','warp-clang.dll'):
                (package/'bin'/name).write_bytes(b'matching-wheel-'+name.encode())
            before = (vendor/'config/extension.toml').read_bytes()
            dest = warp_extension_path(root)
            build_extension(vendor, package, dest)
            self.assertEqual((vendor/'config/extension.toml').read_bytes(), before)
            self.assertEqual((vendor/'warp/old.dll').read_bytes(), b'original')
            self.assertFalse((dest/'warp/old.dll').exists())
            self.assertEqual((dest/'warp/__init__.py').read_bytes(), (package/'__init__.py').read_bytes())
            self.assertIn('version = "'+WARP_VERSION+'"', (dest/'config/extension.toml').read_text())
            self.assertEqual(build_extension(vendor, package, dest), dest)
            arguments = kit_arguments({},root).split()
            self.assertEqual(Path(arguments[arguments.index('--portable-root')+1]), (root/'runtime/kit_state').resolve())
            self.assertEqual(Path(arguments[arguments.index('--ext-path')+1]), dest.resolve())
            (dest/'warp/bin/warp.dll').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                build_extension(vendor, package, dest)

    def test_binary_versions_without_pip_sim_metadata(self):
        from runtime_versions import require_versions
        with tempfile.TemporaryDirectory() as location:
            folder = Path(location)
            (folder/'sim').mkdir()
            (folder/'lab').mkdir()
            (folder/'sim/VERSION').write_text('5.0.0-rc.45+release.23960')
            (folder/'lab/VERSION').write_text('2.2.1')
            settings = dict(isaac_sim_path=str(folder/'sim'), isaac_lab_path=str(folder/'lab'))
            self.assertEqual(require_versions(settings)['isaaclab'], '2.2.1')
            (folder/'sim/VERSION').write_text('5.1.0')
            with self.assertRaises(RuntimeError):
                require_versions(settings)

    def test_urdf_rejects_missing_coupling_and_geometry(self):
        with tempfile.TemporaryDirectory() as location:
            folder = Path(location)
            mesh = folder / 'fixture.stl'
            mesh.write_bytes(b'test fixture')
            joints = ''.join('<joint name="%s" type="revolute">'
                '<limit lower="-1" upper="1" effort="3" velocity="4"/>%s</joint>' %
                (name, '<mimic joint="LHipYawPitch" multiplier="1" offset="0"/>'
                 if name == 'RHipYawPitch' else '') for name in LEG_JOINTS)
            feet = ''.join('<link name="%s"><collision><geometry>'
                          '<mesh filename="fixture.stl"/></geometry></collision></link>' % name
                          for name in ('l_ankle', 'r_ankle'))
            valid = '<robot name="NaoH25V50">' + joints + feet + '</robot>'
            path = folder / 'fixture.urdf'
            path.write_text(valid)
            self.assertEqual(validate_urdf(path)['hip_coupling'], 'physical_mimic_1_to_1')
            path.write_text(valid.replace('multiplier="1"', 'multiplier="-1"'))
            with self.assertRaises(ValueError):
                validate_urdf(path)
            path.write_text(valid)
            mesh.unlink()
            with self.assertRaises(FileNotFoundError):
                validate_urdf(path)
