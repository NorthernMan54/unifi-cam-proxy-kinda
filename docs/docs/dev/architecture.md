# Architecture

The project is being migrated incrementally from the historical `unifi.cams`
package into explicit application boundaries. Existing import paths remain
available while the migration is in progress.

## Boundaries

- `unifi.cli` is the installed command-line entry point.
- `unifi.runtime` assembles a camera and its connection transport.
- `unifi.transport` owns external connection lifecycles.
- `unifi.protocol` contains protocol message types and, eventually, encoding.
- `unifi.events` contains event-domain types independent of cameras.
- `unifi.cameras` is the public namespace for camera implementations.
- `unifi.cams` is the compatibility namespace and current implementation home.

Dependencies should point inward: camera implementations may depend on event
and protocol types, while those types must not import cameras. Transports may
run a camera but should not contain camera-specific behavior. New features
should use the new namespaces; existing modules can be migrated a component at
a time without changing the CLI or downstream imports.

## Next migrations

1. Move the RTSP and Frigate implementations behind `unifi.cameras` modules.
2. Replace manager event dictionaries with the dataclasses in
   `unifi.events.models`.
3. Move protocol dispatch handlers out of the camera base class.
4. Replace the temporary `argparse.Namespace` boundaries with typed settings.
