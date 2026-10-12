CREATE TABLE IF NOT EXISTS people (
  person_id SERIAL PRIMARY KEY, display_name TEXT CHECK (display_name IS NULL OR char_length(display_name) <= 50),
  merged_into INT REFERENCES people(person_id), deleted BOOLEAN NOT NULL DEFAULT FALSE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS images (
  image_id SERIAL PRIMARY KEY, path TEXT UNIQUE NOT NULL, sha256 TEXT UNIQUE NOT NULL,
  width INT, height INT, taken_at TIMESTAMPTZ, gps_lat DOUBLE PRECISION, gps_lon DOUBLE PRECISION,
  caption_template TEXT, presence_template TEXT, activity_text TEXT, scene TEXT,
  tags JSONB NOT NULL DEFAULT '[]', token_map JSONB NOT NULL DEFAULT '[]',
  needs_recaption BOOLEAN NOT NULL DEFAULT FALSE, caption_locked BOOLEAN NOT NULL DEFAULT FALSE,
  status TEXT NOT NULL DEFAULT 'new', error TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS faces (
  face_id SERIAL PRIMARY KEY, image_id INT NOT NULL REFERENCES images(image_id) ON DELETE CASCADE,
  person_id INT REFERENCES people(person_id), bbox JSONB NOT NULL, det_score REAL, blur_var REAL,
  quality_ok BOOLEAN NOT NULL DEFAULT TRUE, assign_sim REAL, assigned_by TEXT
);
ALTER TABLE people ADD COLUMN IF NOT EXISTS profile_face_id INT REFERENCES faces(face_id) ON DELETE SET NULL;
CREATE TABLE IF NOT EXISTS image_people (
  image_id INT NOT NULL REFERENCES images(image_id) ON DELETE CASCADE,
  person_id INT NOT NULL REFERENCES people(person_id), PRIMARY KEY (image_id, person_id)
);
CREATE INDEX IF NOT EXISTS idx_faces_person ON faces(person_id);
CREATE INDEX IF NOT EXISTS idx_faces_image ON faces(image_id);
CREATE INDEX IF NOT EXISTS idx_ip_person ON image_people(person_id);
CREATE INDEX IF NOT EXISTS idx_people_name ON people(lower(display_name));

CREATE TABLE IF NOT EXISTS feedback (
  feedback_id BIGSERIAL PRIMARY KEY,
  feedback_type TEXT NOT NULL CONSTRAINT feedback_feedback_type_check CHECK (feedback_type IN ('caption', 'search', 'classification')),
  image_id INT NOT NULL REFERENCES images(image_id) ON DELETE CASCADE,
  face_id INT REFERENCES faces(face_id) ON DELETE CASCADE,
  wrong_person_id INT REFERENCES people(person_id),
  correct_person_id INT REFERENCES people(person_id),
  corrected_caption TEXT,
  search_query TEXT,
  relevant BOOLEAN,
  rank_position INT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT feedback_check CHECK ((feedback_type = 'caption' AND corrected_caption IS NOT NULL)
      OR (feedback_type = 'search' AND search_query IS NOT NULL AND relevant IS NOT NULL)
      OR (feedback_type = 'classification' AND face_id IS NOT NULL AND correct_person_id IS NOT NULL))
);
CREATE INDEX IF NOT EXISTS idx_feedback_search_image ON feedback(feedback_type, image_id, created_at DESC);
