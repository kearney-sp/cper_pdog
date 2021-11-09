# Workflow for developing training dataset for UAS

### Basic steps
1. For each 100 x 100 m ground sampling site, create a new dataset that keeps all burrows from a 25 x 25 m subsection of the site
1. Create Y x Z m grid across the entire study area
  * This is required for training the CNN. We must ensure that all burrows within a grid cell have been delineated so that the CNN receives complete presence *and* absence training data
2. Randomly iterate through *n* individual grid cells, starting with cells that overlap the 25 x 25 m subsections of ground sampling sites
  * Ensure that the *n* grid cells include at least 5 of the 100 x 100 m ground sampling sites
  * The points remaining in the 25 x 25 m subsections will serve to help 'train the trainers' at the beginning of their demarcations session
3. Demarcate all individual burrows (as points) within each grid cell
  * Skip any points already present in the 25 x 25 m subsections
4. Inspect each point to confirm/reject the burrow and demarcate individual mounds (as polygons) for each confirmed burrow
  * Do this for all points, including the points already present in the 25 x 25 m subsections
5. Validate the burrow (point) demarcations that were added within the 100 x 100 m ground sampling sites (i.e., outside the 25 x 25 m subsections)
    * This will give an 'accuracy' assessment for the individual delineator
6. Validate across all delineators
  * This will give a 'consistency' assessment across delineators

### Remaining questions
* What do we do about inconsistent points/polygons across delineators?
* Are we training the CNN on individual grid cells within each mound?
