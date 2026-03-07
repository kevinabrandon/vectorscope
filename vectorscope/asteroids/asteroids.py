#
# Headless Asteroids game logic adapted for vector rendering.
#

import logging
import math
import random

_game_logger = logging.getLogger('vectorscope.game')

from .badies import Debris, Rock, Saucer
from .ship import Ship
from .soundManager import playSound, playSoundContinuous, stopSound
from .stage import Stage
from .util.vector2d import Vector2d


class VectorRenderer:
    def __init__(self, builder, maxc, aspect_x, max_vectors):
        self.builder = builder
        self.maxc = float(maxc)
        self.aspect_x = float(aspect_x)
        self.max_vectors = int(max_vectors) if max_vectors is not None else 1_000_000
        self.segments_used = 0
        self.cx = self.maxc * 0.5

    def _clamp(self, v):
        if v < 0:
            return 0
        if v > self.maxc:
            return int(self.maxc)
        return int(round(v))

    def _map(self, x, y):
        sx = self.cx + (x - self.cx) * self.aspect_x
        return self._clamp(sx), self._clamp(y)

    def draw_poly(self, sprite, points):
        if self.segments_used >= self.max_vectors:
            return
        if not points or len(points) < 2:
            return
        color = getattr(sprite, "color", (255, 255, 255))
        if isinstance(color, tuple) and len(color) >= 3:
            if color[0] <= 0 and color[1] <= 0 and color[2] <= 0:
                return
        
        # High-intensity for small/fast objects, standard for others
        from .util.vectorsprites import Point
        if isinstance(sprite, Point) or sprite.__class__.__name__ == 'ThrustJet':
            intensity = 1.0
        else:
            intensity = 0.5

        def draw_once():
            x0, y0 = self._map(points[0][0], points[0][1])
            self.builder.move_to(x0, y0, intensity=intensity)
            
            px, py = x0, y0
            for p in points[1:] + [points[0]]: # Iterate all segments including closing
                if self.segments_used >= self.max_vectors:
                    return False
                x1, y1 = self._map(p[0], p[1])
                
                # Detect wrap-around jump (> 50% of screen)
                if abs(x1 - px) > self.maxc * 0.5 or abs(y1 - py) > self.maxc * 0.5:
                    self.builder.move_to(x1, y1, intensity=intensity)
                else:
                    self.builder.line_to(x1, y1)
                    self.segments_used += 1
                
                px, py = x1, y1
            return True

        draw_once()


class Asteroids:
    explodingTtl = 180

    def __init__(self, maxc=2048, aspect_x=0.75, seed=None, num_rocks=3,
                 friendly_fire=False,
                 ship_bullet_speed=None, ship_bullet_ttl=None, ship_max_bullets=None,
                 saucer_bullet_speed=None, saucer_bullet_ttl=None, saucer_max_bullets=None):
        self.maxc = float(maxc)
        self.aspect_x = float(aspect_x)
        self._rng = random.Random(seed)
        # Store CLI overrides — these layer on top of difficulty presets
        self._cli_overrides = {}
        if friendly_fire:
            self._cli_overrides['friendly_fire'] = True
        if num_rocks != 3:
            self._cli_overrides['rocks'] = num_rocks
        for key, val in [('ship_bullet_speed', ship_bullet_speed),
                         ('ship_bullet_ttl', ship_bullet_ttl),
                         ('ship_max_bullets', ship_max_bullets),
                         ('saucer_bullet_speed', saucer_bullet_speed),
                         ('saucer_bullet_ttl', saucer_bullet_ttl),
                         ('saucer_max_bullets', saucer_max_bullets)]:
            if val is not None:
                self._cli_overrides[key] = val
        self.friendly_fire = friendly_fire
        self.ship_bullet_speed = ship_bullet_speed
        self.ship_bullet_ttl = ship_bullet_ttl
        self.ship_max_bullets = ship_max_bullets
        self.saucer_bullet_speed = saucer_bullet_speed
        self.saucer_bullet_ttl = saucer_bullet_ttl
        self.saucer_max_bullets = saucer_max_bullets
        self.stage = Stage("Asteroids", (self.maxc, self.maxc))
        self.attract_mode = True
        self.show_help = False
        self._time = 0.0
        self.initial_rocks = num_rocks
        self.rock_speed = 1.0
        self.current_difficulty = 'hard'
        self.difficulty_presets = {
            'easy':   dict(rocks=1, lives=5, friendly_fire=False, rock_speed=0.8,
                           saucer_bullet_speed=None, saucer_bullet_ttl=None, saucer_max_bullets=None,
                           ship_bullet_speed=None, ship_bullet_ttl=None, ship_max_bullets=None),
            'medium': dict(rocks=3, lives=4, friendly_fire=False, rock_speed=1.0,
                           saucer_bullet_speed=10, saucer_bullet_ttl=90, saucer_max_bullets=2,
                           ship_bullet_speed=None, ship_bullet_ttl=None, ship_max_bullets=None),
            'hard':   dict(rocks=4, lives=3, friendly_fire=True, rock_speed=1.2,
                           saucer_bullet_speed=15, saucer_bullet_ttl=120, saucer_max_bullets=3,
                           ship_bullet_speed=None, ship_bullet_ttl=None, ship_max_bullets=None),
        }
        
        self._level_num = 1
        self.ship = None
        self.reset_attract()

    def _log(self, category, msg, level=logging.INFO):
        cat = category.strip('[]').lower()
        _game_logger.log(level, msg, extra={'category': cat})

    def _log_score(self, increase, reason):
        self.score += increase
        self._log("[score]", f"+{increase} points ({reason}). Total Score: {self.score}")

    def _get_rock_counts(self):
        l, m, s = 0, 0, 0
        for r in self.rockList:
            if r.rockType == Rock.largeRockType: l += 1
            elif r.rockType == Rock.mediumRockType: m += 1
            elif r.rockType == Rock.smallRockType: s += 1
        return l, m, s

    def reset_attract(self):
        self.stage.clear()
        self.gameState = "playing"
        self.score = 0
        self.lives = 3
        self.rockList = []
        p = self.difficulty_presets[self.current_difficulty]
        self.numRocks = p['rocks']
        self._level_num = 1
        self._log("[Level]", f"Level 1 started. Rocks: {self.numRocks}, Score: {self.score}, Lives: {self.lives}")
        self.nextLife = 10000
        self.saucer = None
        self._orphaned_bullets = []
        self._saucer_spawn_timer = self._rng.uniform(6.0, 12.0)
        self.explodingCount = 0.0
        self.createRocks(self.numRocks)
        self.createNewShip()
        self._reset_ai()

    def start_game(self, difficulty):
        """Start a new game with the given difficulty preset, then apply CLI overrides."""
        p = dict(self.difficulty_presets[difficulty])
        # CLI args override preset values
        p.update(self._cli_overrides)
        self.initial_rocks = p['rocks']
        self.friendly_fire = p['friendly_fire']
        self.rock_speed = p['rock_speed']
        self.ship_bullet_speed = p['ship_bullet_speed']
        self.ship_bullet_ttl = p['ship_bullet_ttl']
        self.ship_max_bullets = p['ship_max_bullets']
        self.saucer_bullet_speed = p['saucer_bullet_speed']
        self.saucer_bullet_ttl = p['saucer_bullet_ttl']
        self.saucer_max_bullets = p['saucer_max_bullets']
        self.current_difficulty = difficulty
        self.attract_mode = False
        self.show_help = False
        self.reset_attract()
        self.lives = p['lives']
        self.createNewShip()

    def continue_game(self):
        """Continue after game over, keeping score and rocks."""
        p = self.difficulty_presets[self.current_difficulty]
        self.lives = p['lives']
        self.gameState = "playing"
        self.createNewShip()

    def _reset_ai(self):
        self._ai_turn_timer = 0.0
        self._ai_thrust_timer = 0.0
        self._ai_fire_timer = 0.0
        self._ai_turn_dir = 1
        self._ai_thrusting = False

    def find_safe_position(self):
        """Find the position farthest from all rocks and saucers."""
        threats = [r.position for r in self.rockList]
        if self.saucer is not None:
            threats.append(self.saucer.position)
        if not threats:
            return self.maxc / 2, self.maxc / 2
        best_x, best_y, best_dist = self.maxc / 2, self.maxc / 2, 0
        margin = self.maxc * 0.1
        for _ in range(50):
            x = self._rng.uniform(margin, self.maxc - margin)
            y = self._rng.uniform(margin, self.maxc - margin)
            min_dist = min(math.hypot(x - t.x, y - t.y) for t in threats)
            if min_dist > best_dist:
                best_x, best_y, best_dist = x, y, min_dist
        return best_x, best_y

    def createNewShip(self):
        if self.ship:
            for debris in list(self.ship.shipDebrisList):
                self.stage.removeSprite(debris)
            self.ship.shipDebrisList = []
            self.stage.removeSprite(self.ship)
            self.stage.removeSprite(self.ship.thrustJet)
        self.ship = Ship(self.stage)
        safe_x, safe_y = self.find_safe_position()
        self.ship.position.x = safe_x
        self.ship.position.y = safe_y
        self.ship.thrustJet.position.x = safe_x
        self.ship.thrustJet.position.y = safe_y
        if self.ship_bullet_speed is not None:
            self.ship.bulletVelocity = self.ship_bullet_speed
        if self.ship_bullet_ttl is not None:
            self.ship.bulletTtl = self.ship_bullet_ttl
        if self.ship_max_bullets is not None:
            self.ship.maxBullets = self.ship_max_bullets
        self.stage.addSprite(self.ship.thrustJet)
        self.stage.addSprite(self.ship)

    def _apply_rock_speed(self, rock):
        rock.heading.x *= self.rock_speed
        rock.heading.y *= self.rock_speed

    def createRocks(self, numRocks):
        margin = self.maxc * 0.1
        min_ship_dist = self.maxc * 0.25
        for _ in range(0, numRocks):
            for _ in range(50):
                x = self._rng.uniform(margin, self.maxc - margin)
                y = self._rng.uniform(margin, self.maxc - margin)
                if not self.ship or math.hypot(x - self.ship.position.x, y - self.ship.position.y) > min_ship_dist:
                    break
            position = Vector2d(x, y)
            newRock = Rock(self.stage, position, Rock.largeRockType)
            self._apply_rock_speed(newRock)
            self.stage.addSprite(newRock)
            self.rockList.append(newRock)

    def levelUp(self):
        self._level_num += 1
        self.numRocks += 1
        self._log("[Level]", f"Level {self._level_num} started. Rocks: {self.numRocks}, Score: {self.score}, Lives: {self.lives}")
        self.createRocks(self.numRocks)

    def attractControl(self, dt):
        if not self.ship:
            return
        step = dt * 60.0
        self._ai_turn_timer -= dt
        if self._ai_turn_timer <= 0.0:
            self._ai_turn_timer = self._rng.uniform(0.25, 1.0)
            self._ai_turn_dir = -1 if self._rng.random() < 0.5 else 1
        if self._ai_turn_dir < 0:
            self.ship.rotateLeft(step)
        else:
            self.ship.rotateRight(step)

        self._ai_thrust_timer -= dt
        if self._ai_thrust_timer <= 0.0:
            self._ai_thrust_timer = self._rng.uniform(0.35, 1.2)
            self._ai_thrusting = not self._ai_thrusting
        if self._ai_thrusting:
            self.ship.increaseThrust(step)
            self.ship.thrustJet.accelerating = True
        else:
            self.ship.thrustJet.accelerating = False

        self._ai_fire_timer -= dt
        if self._ai_fire_timer <= 0.0:
            self._ai_fire_timer = self._rng.uniform(0.2, 0.6)
            self.ship.fireBullet()

    def doSaucerLogic(self, dt):
        if self.saucer is not None and self.saucer.laps >= 2:
            self.killSaucer(reason="max laps")

        if self.saucer is None:
            self._saucer_spawn_timer -= dt
            if self._saucer_spawn_timer <= 0.0:
                rand_val = self._rng.random()
                saucer_type = Saucer.smallSaucerType if rand_val <= 0.4 else Saucer.largeSaucerType
                self.saucer = Saucer(self.stage, saucer_type, self.ship)
                if self.saucer_bullet_speed is not None:
                    self.saucer.bulletVelocity = self.saucer_bullet_speed
                if self.saucer_bullet_ttl is not None:
                    self.saucer.bulletTtl = [self.saucer_bullet_ttl, self.saucer_bullet_ttl]
                if self.saucer_max_bullets is not None:
                    self.saucer.maxBullets = self.saucer_max_bullets
                self.stage.addSprite(self.saucer)
                self._saucer_spawn_timer = self._rng.uniform(6.0, 12.0)

    def exploding(self, dt):
        self.explodingCount += dt * 60.0
        if self.explodingCount > self.explodingTtl:
            for debris in list(self.ship.shipDebrisList):
                self.stage.removeSprite(debris)
            self.ship.shipDebrisList = []

            if self.lives <= 0:
                if self.attract_mode:
                    self.reset_attract()
                else:
                    self.gameState = "gameover"
                    self.ship = None
            else:
                self.gameState = "playing"
                self.createNewShip()

    def checkCollisions(self):
        if not self.ship:
            return
        shipHit, saucerHit = False, False

        for rock in list(self.rockList):
            rockHit = False

            if not self.ship.inHyperSpace and rock.collidesWith(self.ship):
                p = rock.checkPolygonCollision(self.ship)
                if p is not None:
                    shipHit = True
                    rockHit = True

            if self.saucer is not None:
                if rock.collidesWith(self.saucer):
                    saucerHit = True
                    rockHit = True

                if self.saucer.bulletCollision(rock):
                    rockHit = True

                if self.ship.bulletCollision(self.saucer):
                    saucerHit = True
                    self._log_score(self.saucer.scoreValue, f"{'Small' if self.saucer.saucerType == Saucer.smallSaucerType else 'Large'} saucer destroyed")
                    self._log("[bullet]", "Ship bullet hit saucer.")
                    self.createDebris(self.saucer)
                    self.killSaucer(reason="shot")
            if self.ship.bulletCollision(rock):
                rockHit = True
                self._log("[bullet]", "Ship bullet hit rock.")

            if rockHit:
                self.rockList.remove(rock)
                self.stage.removeSprite(rock)

                size_map = {Rock.largeRockType: "Large", Rock.mediumRockType: "Medium", Rock.smallRockType: "Small"}
                old_size = size_map.get(rock.rockType, "Unknown")
                created = 0

                if rock.rockType == Rock.largeRockType:
                    playSound("explode1")
                    newRockType = Rock.mediumRockType
                    self._log_score(50, f"{old_size} rock destroyed")
                    created = 2
                elif rock.rockType == Rock.mediumRockType:
                    playSound("explode2")
                    newRockType = Rock.smallRockType
                    self._log_score(100, f"{old_size} rock destroyed")
                    created = 2
                else:
                    playSound("explode3")
                    self._log_score(200, f"{old_size} rock destroyed")

                if created > 0:
                    for _ in range(0, created):
                        position = Vector2d(rock.position.x, rock.position.y)
                        newRock = Rock(self.stage, position, newRockType)
                        self._apply_rock_speed(newRock)
                        self.stage.addSprite(newRock)
                        self.rockList.append(newRock)

                l, m, s = self._get_rock_counts()
                self._log("[rocks]", f"{old_size} rock exploded. Created {created} rocks. Field: {l}L, {m}M, {s}S")
                self.createDebris(rock)

        if self.saucer is not None:
            if not self.ship.inHyperSpace:
                if self.saucer.bulletCollision(self.ship):
                    shipHit = True

                if self.saucer.collidesWith(self.ship):
                    shipHit = True
                    saucerHit = True

            if saucerHit:
                self.createDebris(self.saucer)
                self.killSaucer(reason="collision")

        # Check orphaned saucer bullets (saucer dead, bullets still in flight)
        for bullet in list(self._orphaned_bullets):
            if bullet.ttl <= 0:
                self._orphaned_bullets.remove(bullet)
                continue
            # Orphaned bullets can still hit rocks
            for rock in list(self.rockList):
                if rock.collidesWith(bullet):
                    bullet.ttl = 0
                    self._orphaned_bullets.remove(bullet)
                    self.rockList.remove(rock)
                    self.stage.removeSprite(rock)
                    if rock.rockType == Rock.largeRockType:
                        playSound("explode1")
                        newRockType = Rock.mediumRockType
                    elif rock.rockType == Rock.mediumRockType:
                        playSound("explode2")
                        newRockType = Rock.smallRockType
                    else:
                        playSound("explode3")
                    if rock.rockType != Rock.smallRockType:
                        for _ in range(2):
                            position = Vector2d(rock.position.x, rock.position.y)
                            newRock = Rock(self.stage, position, newRockType)
                            self._apply_rock_speed(newRock)
                            self.stage.addSprite(newRock)
                            self.rockList.append(newRock)
                    self.createDebris(rock)
                    break
            # Orphaned bullets can still hit the ship
            if not shipHit and not self.ship.inHyperSpace:
                if bullet in self._orphaned_bullets and self.ship.collidesWith(bullet):
                    shipHit = True
                    bullet.ttl = 0
                    self._orphaned_bullets.remove(bullet)

        if not shipHit and self.friendly_fire and not self.ship.inHyperSpace:
            for bullet in self.ship.bullets:
                if bullet.ttl > 0 and bullet.ttl < 80 and self.ship.collidesWith(bullet):
                    shipHit = True
                    bullet.ttl = 0
                    break

        if shipHit:
            self.killShip()

    def checkPostDeathCollisions(self):
        """Check ship's in-flight bullets against rocks/saucers while ship is dead."""
        if not self.ship:
            return
        for rock in list(self.rockList):
            if self.ship.bulletCollision(rock):
                self.rockList.remove(rock)
                self.stage.removeSprite(rock)
                
                size_map = {Rock.largeRockType: "Large", Rock.mediumRockType: "Medium", Rock.smallRockType: "Small"}
                old_size = size_map.get(rock.rockType, "Unknown")
                created = 0

                if rock.rockType == Rock.largeRockType:
                    playSound("explode1")
                    newRockType = Rock.mediumRockType
                    self._log_score(50, f"{old_size} rock destroyed (post-death)")
                    created = 2
                elif rock.rockType == Rock.mediumRockType:
                    playSound("explode2")
                    newRockType = Rock.smallRockType
                    self._log_score(100, f"{old_size} rock destroyed (post-death)")
                    created = 2
                else:
                    playSound("explode3")
                    self._log_score(200, f"{old_size} rock destroyed (post-death)")

                if created > 0:
                    for _ in range(2):
                        position = Vector2d(rock.position.x, rock.position.y)
                        newRock = Rock(self.stage, position, newRockType)
                        self._apply_rock_speed(newRock)
                        self.stage.addSprite(newRock)
                        self.rockList.append(newRock)
                
                l, m, s = self._get_rock_counts()
                self._log("[rocks]", f"{old_size} rock exploded (post-death). Created {created} rocks. Field: {l}L, {m}M, {s}S")
                self.createDebris(rock)

        if self.saucer is not None:
            if self.ship.bulletCollision(self.saucer):
                self._log_score(self.saucer.scoreValue, f"{'Small' if self.saucer.saucerType == Saucer.smallSaucerType else 'Large'} saucer destroyed (post-death)")
                self.createDebris(self.saucer)
                self.killSaucer(reason="shot post-death")

    def killShip(self):
        stopSound("thrust")
        playSound("explode2")
        self.explodingCount = 0.0
        self.lives -= 1
        self.stage.removeSprite(self.ship)
        self.stage.removeSprite(self.ship.thrustJet)
        self.gameState = "exploding"
        self.ship.explode()

    def killSaucer(self, reason=None):
        stopSound("lsaucer")
        stopSound("ssaucer")
        playSound("explode2")
        if reason:
            self._log("[saucer]", f"Saucer killed: {reason}")
        self.stage.removeSprite(self.saucer)
        # Keep any in-flight bullets so they can still hit the player
        for bullet in self.saucer.bullets:
            self._orphaned_bullets.append(bullet)
        self.saucer = None

    def createDebris(self, sprite):
        for _ in range(0, 25):
            position = Vector2d(sprite.position.x, sprite.position.y)
            debris = Debris(position, self.stage)
            self.stage.addSprite(debris)

    def checkScore(self):
        if self.attract_mode:
            return
        if self.score > 0 and self.score > self.nextLife:
            playSound("extralife")
            self.nextLife += 10000
            self.lives += 1

    def step(self, dt, builder, max_vectors):
        if dt < 0:
            dt = 0.0
        if dt > 0.2:
            dt = 0.2
        self._time += dt

        self.stage.moveSprites(dt)

        renderer = VectorRenderer(builder, self.maxc, self.aspect_x, max_vectors)
        self.stage.drawSprites(renderer.draw_poly)

        self.doSaucerLogic(dt)
        self.checkScore()

        if self.gameState == "playing":
            if self.attract_mode:
                self.attractControl(dt)
            self.checkCollisions()
            if len(self.rockList) == 0:
                self.levelUp()
        elif self.gameState == "exploding":
            self.checkPostDeathCollisions()
            self.exploding(dt)

        self.render_ui(builder)

    def render_ui(self, builder):
        def map_x(x):
            cx = self.maxc * 0.5
            return int(round(cx + (x - cx) * self.aspect_x))

        score_str = f"{self.score:06d}"
        builder.text_at(map_x(80), int(self.maxc - 120), score_str, size=4.5, rot=0, intensity=0.5)

        lives_str = f"{max(0, self.lives)}"
        # Top right
        builder.text_at(map_x(self.maxc - 150), int(self.maxc - 120), lives_str, size=4.5, rot=0, intensity=0.5)

        center_x = self.maxc * 0.5

        if self.show_help:
            sz = 5
            gap = int(self.maxc * 0.1)
            lines = ["0 DEMO", "1 EASY", "2 MEDIUM", "3 HARD"]
            top_y = int(self.maxc * 0.5 + gap * 1.5)
            for i, line in enumerate(lines):
                builder.text_at_centered(center_x, top_y - i * gap, line, size=sz, rot=0, intensity=0.5)
        elif self.attract_mode:
            title = "ASTEROIDS"
            title_y = int(self.maxc * 0.78)
            builder.text_at_centered(center_x, title_y, title, size=8, rot=0, intensity=0.5)

            if int(self._time * 1.5) % 2 == 0:
                builder.text_at_centered(center_x, int(self.maxc * 0.65), "PRESS ? FOR HELP", size=4.5, rot=0, intensity=0.5)
        elif self.gameState == "gameover":
            builder.text_at_centered(center_x, int(self.maxc * 0.55), "GAME OVER", size=7, rot=0, intensity=0.5)
            if int(self._time * 1.5) % 2 == 0:
                builder.text_at_centered(center_x, int(self.maxc * 0.4), "C TO CONT", size=4.5, rot=0, intensity=0.5)
