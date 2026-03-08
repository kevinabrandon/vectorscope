#    Copyright (C) 2008  Nick Redshaw
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#    Copyright (C) 2008  Nick Redshaw
#

import math
import random

from .shooter import Shooter
from .soundManager import playSound, playSoundContinuous, stopSound
from .util.vector2d import Vector2d
from .util.vectorsprites import Point, VectorSprite

# Four different shape of rock each of which can be small, medium or large.
# Smaller rocks are faster.
class Rock(VectorSprite):
    
    # indexes into the tuples below
    largeRockType = 0
    mediumRockType = 1
    smallRockType = 2   
    
    velocities = (1.5, 4.0, 6.0)    
    scales = (15.0, 9.0, 3.6)

    # tracks the last rock shape to be generated
    rockShape = 1    
    
    # Create the rock polygon to the given scale
    def __init__(self, stage, position, rockType):
        
        scale = Rock.scales[rockType]
        velocity = Rock.velocities[rockType]                
        heading = Vector2d(random.uniform(-velocity, velocity), random.uniform(-velocity, velocity))
        
        # Ensure that the rocks don't just sit there or move along regular lines
        if heading.x == 0:
            heading.x = 0.1
        
        if heading.y == 0:
            heading.y = 0.1
                        
        self.rockType = rockType  
        pointlist = self.createPointList()
        newPointList = [self.scale(point, scale) for point in pointlist]        
        VectorSprite.__init__(self, position, heading, newPointList)
                
    
    # Create different rock type pointlists    
    def createPointList(self):
        
        if (Rock.rockShape == 1):
            pointlist = [(-4,-12), (6,-12), (13, -4), (13, 5), (6, 13), (0,13), (0,4),\
                     (-8,13), (-15, 4), (-7,1), (-15,-3)]
 
        elif (Rock.rockShape == 2):
            pointlist = [(-6,-12), (1,-5), (8, -12), (15, -5), (12,0), (15,6), (5,13),\
                         (-7,13), (-14,7), (-14,-5)]
            
        elif (Rock.rockShape == 3):
            pointlist = [(-7,-12), (1,-9), (8,-12), (15,-5), (8,-3), (15,4), (8,12),\
                         (-3,10), (-6,12), (-14,7), (-10,0), (-14,-5)]            

        elif (Rock.rockShape == 4):
            pointlist = [(-7,-11), (3,-11), (13,-5), (13,-2), (2,2), (13,8), (6,14),\
                         (2,10), (-7,14), (-15,5), (-15,-5), (-5,-5), (-7,-11)]

        Rock.rockShape += 1
        if (Rock.rockShape == 5):
            Rock.rockShape = 1

        return pointlist
    
    # Spin the rock when it moves
    def move(self, step=1.0):
        VectorSprite.move(self, step)
        
        # Original Asteroid didn't have spinning rocks but they look nicer
        self.angle += 1 * step
    
    
#    def destroyed(self):
        

class Debris(Point):

    def __init__(self, position, stage, velocity=1.5, ttl=50):
        heading = Vector2d(random.uniform(-velocity, velocity), random.uniform(-velocity, velocity))
        Point.__init__(self, position, heading, stage)
        self.ttl = ttl

    def move(self, step=1.0):
        Point.move(self, step)
        r, g, b = self.color
        decay = int(5 * step)
        r -= decay
        g -= decay
        b -= decay
        self.color = (r, g, b)


class LineDebris(Debris):
    """Minimal 2-point line segment debris — for small rock explosions."""
    pointlist = [(-3, 0), (3, 0)]

    def __init__(self, position, stage, velocity=1.5, ttl=30):
        Debris.__init__(self, position, stage, velocity=velocity, ttl=ttl)
        

# Flying saucer, shoots at player
class Saucer(Shooter):
    
    # indexes into the tuples below
    largeSaucerType = 0
    smallSaucerType = 1

    velocities = (1.7, 3.0)    
    scales = (12.0, 8.0)
    scores = (500, 1000)
    pointlist = [(-9,0), (-3,-3), (-2,-6), (-2,-6), (2,-6), (3,-3), (9,0), (-9,0), (-3,4), (3,4), (9,0)]
    maxBullets = 1
    bulletTtl = [60, 90]
    bulletVelocity = 5  
    
    def __init__(self, stage, saucerType, ship):
        mid_lo, mid_hi = 0.25, 0.75
        v = self.velocities[saucerType]
        side = random.choice(('left', 'right', 'bottom'))

        if side == 'left':
            x = 0.0
            y = random.uniform(stage.height * mid_lo, stage.height * mid_hi)
            vx, vy = v, 0.0
        elif side == 'right':
            x = float(stage.width)
            y = random.uniform(stage.height * mid_lo, stage.height * mid_hi)
            vx, vy = -v, 0.0
        else:  # bottom
            x = random.uniform(stage.width * mid_lo, stage.width * mid_hi)
            y = 0.0
            vx, vy = 0.0, v

        position = Vector2d(x, y)
        heading = Vector2d(vx, vy)
        self.side = side
        self.saucerType = saucerType
        self.ship = ship
        self.scoreValue = self.scores[saucerType]
        stopSound("ssaucer")
        stopSound("lsaucer")
        if saucerType == self.largeSaucerType:
            playSoundContinuous("lsaucer")
        else:
            playSoundContinuous("ssaucer")
        self.laps = 0
        self._last_primary = x if side in ('left', 'right') else y
        self._fire_cooldown = 0.0

        # Scale the shape and create the VectorSprite
        flipped = [(px, -py) for px, py in self.pointlist]
        newPointList = [self.scale(point, self.scales[saucerType]) for point in flipped]
        Shooter.__init__(self, position, heading, newPointList, stage)

    def move(self, step=1.0):
        Shooter.move(self, step)

        if self.side in ('left', 'right'):
            # Vertical drift in the horizontal middle zone
            if self.stage.width * 0.33 < self.position.x < self.stage.width * 0.66:
                self.heading.y = self.heading.x
            else:
                self.heading.y = 0.0
            # Lap detection: x wraps opposite to travel direction
            cur = self.position.x
            if self.side == 'left' and self._last_primary > cur:
                self._last_primary = 0.0
                self.laps += 1
            elif self.side == 'right' and self._last_primary < cur:
                self._last_primary = float(self.stage.width)
                self.laps += 1
            else:
                self._last_primary = cur
        else:  # bottom, moving up
            # Horizontal drift in the vertical middle zone
            if self.stage.height * 0.33 < self.position.y < self.stage.height * 0.66:
                self.heading.x = self.heading.y
            else:
                self.heading.x = 0.0
            # Lap detection: y wraps from top back to bottom
            cur = self.position.y
            if self._last_primary > cur:
                self._last_primary = 0.0
                self.laps += 1
            else:
                self._last_primary = cur

        self._fire_cooldown -= step
        if self._fire_cooldown <= 0:
            self.fireBullet()
                
    # Set the bullet velocity and create the bullet
    def fireBullet(self):
        if self.ship is not None:            
            dx = self.ship.position.x - self.position.x
            dy = self.ship.position.y - self.position.y
            mag = math.sqrt(dx * dx + dy * dy)
            heading = Vector2d(self.bulletVelocity * (dx/mag), self.bulletVelocity * (dy/mag))
            position = Vector2d(self.position.x, self.position.y)          
            shotFired = Shooter.fireBullet(self, heading, self.bulletTtl[self.saucerType], self.bulletVelocity)
            if shotFired:
                self._fire_cooldown = 30.0
                playSound("sfire")
            
# end    
