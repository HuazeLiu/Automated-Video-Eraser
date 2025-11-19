## Third-Party Dependencies

- `OSTrack/`: Official single-object tracking implementation kept outside version control to avoid pushing a large upstream repository.

Recommended workflow:
1. Clone or extract the official release into `third_party/OSTrack` whenever you need to run the tracker.
2. Record the exact commit/tag (or add a git submodule) if experiments depend on a fixed version.
3. Share your custom changes via a fork or patch file rather than modifying the pristine vendor copy.*** End Patch

