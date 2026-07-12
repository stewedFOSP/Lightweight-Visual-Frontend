# Lightweight-Visual-Frontend

This repository includes a minimal placeholder GitHub Pages site for anonymous review.

## GitHub Pages

The static site lives in `docs/` so it can be published from the `main` branch and remain compatible with `anonymous.4open.science`, which only supports Pages content defined in the same branch as the repository mirror.

After pushing the repository, enable GitHub Pages in the repository settings:

1. Open `Settings -> Pages`
2. Set `Source` to `Deploy from a branch`
3. Set `Branch` to `main`
4. Set the folder to `/docs`

The public Pages URL should then be:

`https://stewedfosp.github.io/Lightweight-Visual-Frontend/`

## Anonymous mirror

Once the repository is public and the page is reachable, submit the repository URL to:

`https://anonymous.4open.science/anonymize`

That service can mirror the repository and expose an anonymized Pages site because the page assets are stored in the same branch.
